"""Two-epoch extension of the proven Phase 1 MPS Whisper LoRA recipe.

Save complete resumable state at optimizer boundaries near every half epoch.
Existing Phase 1 models and training code are never overwritten.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'phase1_whisper_anchor'))
from training.train_whisper_lora import AudioRows, collate, verify_modules, TARGETS

def train(config, manifest, output, resume=None, max_batches=None, stop_after=None, max_runtime=None):
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f'refusing to overwrite training run: {output}')
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    seed = int(config['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == 'cuda': torch.cuda.manual_seed_all(seed)
    if device == 'mps':
        torch.mps.manual_seed(seed)
        torch.mps.set_per_process_memory_fraction(.9)
    frame = pd.read_csv(manifest, sep='\t', keep_default_na=False)
    if manifest.name == 'final_train.tsv':
        assert frame.domain.value_counts().to_dict() == {'jember': 1309, 'competition': 372}
    holdout = pd.read_csv(ROOT / 'phase1_whisper_anchor/data/jember_holdout.tsv', sep='\t', keep_default_na=False)
    assert not set(frame.clip_id) & set(holdout.clip_id), 'Jember holdout leakage'
    processor = WhisperProcessor.from_pretrained(config['base_model'], local_files_only=True)
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(config['base_model'], torch_dtype=dtype, local_files_only=True)
    model.config.apply_spec_augment = bool(config.get('spec_augment', False))
    model.config.mask_time_prob = .05 if model.config.apply_spec_augment else 0.
    model.config.mask_feature_prob = .05 if model.config.apply_spec_augment else 0.
    verify_modules(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(r=int(config['lora_r']), lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']), target_modules=list(TARGETS), bias='none')).to(device)
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'device={device} trainable={active} rows={len(frame)} domains={frame.domain.value_counts().to_dict()}', flush=True)
    dataset = AudioRows(frame, processor, config['language'], config)
    weights = torch.tensor([float(config['competition_weight']) if d == 'competition' else 1. for d in frame.domain], dtype=torch.double)
    draws = int(weights.sum())
    micro = int(config['micro_batch_size'])
    batches = math.ceil(draws / micro)
    if max_batches: batches = min(batches, max_batches); draws = batches * micro
    accumulation = math.ceil(int(config['effective_batch_size']) / micro)
    updates_per_epoch = math.ceil(batches / accumulation)
    total_updates = updates_per_epoch * int(config['epochs'])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']), foreach=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, round(total_updates * float(config['warmup_fraction'])), total_updates)
    generator = torch.Generator().manual_seed(seed)
    state = None
    if resume:
        state = torch.load(resume / 'training_state.pt', map_location='cpu', weights_only=False)
        assert state['config'] == config and state['manifest'] == str(manifest.resolve())
        from peft import load_peft_weights, set_peft_model_state_dict
        set_peft_model_state_dict(model, load_peft_weights(str(resume), device='cpu'))
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        generator.set_state(state['sampler_rng'])
        print(f'resuming {resume}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'provenance.json').write_text(json.dumps({'manifest': str(manifest.resolve()), 'domains': frame.domain.value_counts().to_dict(),
        'draws_per_epoch': draws, 'optimizer_updates_per_epoch': updates_per_epoch, 'manifest_sha256': __import__('hashlib').sha256(manifest.read_bytes()).hexdigest()}, indent=2) + '\n')
    step = state['step'] if state else 0
    start_epoch = state['epoch'] if state else 0
    model.train(); optimizer.zero_grad(set_to_none=True)
    if state and state.get('gradients'):
        named = dict(model.named_parameters())
        for name, gradient in state['gradients'].items():
            if name not in named or not named[name].requires_grad:
                raise ValueError(f'unknown accumulated gradient: {name}')
            if named[name].shape != gradient.shape:
                raise ValueError(f'gradient shape mismatch: {name}')
            named[name].grad = gradient.to(device=device, dtype=named[name].dtype)
        if state['accumulated_microbatches'] != state['next_batch'] % accumulation:
            raise ValueError('saved gradient accumulation position is inconsistent')
        print(f"restored {len(state['gradients'])} accumulated gradient tensors for {state['accumulated_microbatches']} microbatches", flush=True)
    run_started = time.monotonic()
    prior_elapsed = state.get('total_elapsed_s', 0) if state else 0
    stop_requested = {'value': False}
    def request_stop(signum, frame): stop_requested['value'] = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    for epoch in range(start_epoch, int(config['epochs'])):
        continuing = state is not None and epoch == state['epoch'] and state['indices'] is not None
        indices = state['indices'] if continuing else torch.multinomial(weights, draws, replacement=True, generator=generator).tolist()
        offset = state['next_batch'] if continuing else 0
        loss_sum = state['loss_sum'] if continuing else 0.
        epoch_prior_elapsed = state['epoch_elapsed_s'] if continuing else 0.
        loader = DataLoader(dataset, batch_size=micro, sampler=indices[offset*micro:], collate_fn=collate,
            num_workers=0 if device == 'mps' else 4, pin_memory=device == 'cuda')
        iterator = iter(loader)
        if state is not None and epoch == state['epoch']:
            random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng'])
            if device == 'mps': torch.mps.set_rng_state(state['device_rng'])
            elif device == 'cuda': torch.cuda.set_rng_state_all(state['device_rng'])
        started = time.monotonic()
        half_batch = min(batches, math.ceil(math.ceil(batches / 2) / accumulation) * accumulation)
        for batch_index, batch in enumerate(iterator, offset):
            batch = {k: v.to(device, non_blocking=device == 'cuda') for k, v in batch.items()}
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if dtype == torch.bfloat16 else nullcontext()
            with amp: raw_loss = model(**batch).loss
            raw_value = float(raw_loss.detach())
            if not math.isfinite(raw_value): raise FloatingPointError(f'nonfinite loss at epoch {epoch+1}, batch {batch_index+1}')
            (raw_loss / accumulation).backward(); loss_sum += raw_value
            boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == batches
            if boundary:
                norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), float(config['gradient_clip']), error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
            if (batch_index + 1) % 10 == 0 or batch_index + 1 == batches:
                progress = {'epoch': epoch+1, 'batch': batch_index+1, 'batches_per_epoch': batches, 'optimizer_step': step,
                    'mean_loss': loss_sum/(batch_index+1), 'elapsed_s': time.monotonic()-started+epoch_prior_elapsed}
                (output/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
            if (batch_index + 1) % 50 == 0 or batch_index + 1 == batches:
                elapsed = time.monotonic() - started + epoch_prior_elapsed
                completed = batch_index + 1
                print(f'epoch={epoch+1} batch={completed}/{batches} loss={raw_value:.5f} mean={loss_sum/completed:.5f} lr={scheduler.get_last_lr()[0]:.8f} elapsed_s={elapsed:.1f}', flush=True)
            if max_runtime and time.monotonic()-run_started >= max_runtime: stop_requested['value'] = True
            if (stop_after == 'half' and batch_index+1 == half_batch) or (stop_after == 'full' and batch_index+1 == batches): stop_requested['value'] = True
            if batch_index + 1 in {half_batch, batches} or (batch_index+1)%256 == 0 or stop_requested['value']:
                name = 'half' if batch_index+1 == half_batch and half_batch != batches else 'full' if batch_index+1 == batches else 'batch_'+str(batch_index+1).zfill(6)
                checkpoint = output / f"epoch_{epoch+1}_{name}"
                if checkpoint.exists(): raise FileExistsError(checkpoint)
                checkpoint.mkdir()
                model.save_pretrained(checkpoint, safe_serialization=True); processor.save_pretrained(checkpoint)
                metrics = {'epoch': epoch+1, 'batch': batch_index+1, 'optimizer_step': step,
                    'epoch_fraction': epoch+(batch_index+1)/batches, 'mean_loss': loss_sum/(batch_index+1),
                    'elapsed_s': time.monotonic()-started+epoch_prior_elapsed, 'total_elapsed_s': time.monotonic()-run_started+prior_elapsed,
                    'learning_rate': scheduler.get_last_lr()[0]}
                (checkpoint / 'train_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
                saved = {'config': config, 'manifest': str(manifest.resolve()), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch, 'next_batch': batch_index+1, 'indices': indices,
                    'sampler_rng': generator.get_state(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(), 'device_rng': torch.mps.get_rng_state() if device == 'mps' else torch.cuda.get_rng_state_all() if device == 'cuda' else None,
                    'step': step, 'loss_sum': loss_sum, 'epoch_elapsed_s': metrics['elapsed_s'], 'total_elapsed_s': metrics['total_elapsed_s'], 'gradients': {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}, 'accumulated_microbatches': (batch_index+1)%accumulation}
                # Epoch-complete resumes begin a new draw sequence, with no stale loss.
                if batch_index+1 == batches:
                    saved.update(epoch=epoch+1, next_batch=0, indices=None, loss_sum=0., epoch_elapsed_s=0., gradients={}, accumulated_microbatches=0)
                temporary = checkpoint / 'training_state.tmp'
                torch.save(saved, temporary); temporary.replace(checkpoint / 'training_state.pt')
                (output/'latest_checkpoint.txt').write_text(str(checkpoint.resolve())+'\n')
                print(f'SAVED {checkpoint} epoch_fraction={metrics["epoch_fraction"]:.6f}', flush=True)
                if stop_requested['value']:
                    print('GRACEFUL STOP', checkpoint, flush=True); return
        state = None
    print(f'TRAINING COMPLETE {output}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--stop-after', choices=['half','full'])
    parser.add_argument('--max-runtime', type=float)
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), args.manifest, args.output, args.resume, args.max_batches, args.stop_after, args.max_runtime)
