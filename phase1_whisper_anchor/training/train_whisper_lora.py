"""A100 LoRA trainer. Saves adapters every half epoch for fold-wise selection."""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0,str(HERE))
from inference.audio import load_audio, RATE
from training.text import to_reference_style

TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")

def split_long_training(wave: np.ndarray, text: str) -> tuple[tuple[np.ndarray,str],tuple[np.ndarray,str]]:
    """Approximate mid-silence split for labeled 30-40 s clips.

    Reference text has no word timestamps, so divide words by relative time.
    This is explicitly an approximation, used only for the 29 long dev clips.
    """
    if len(wave)<=30*RATE: raise ValueError("split is only for long clips")
    midpoint=len(wave)//2; radius=4*RATE; step=RATE//10
    candidates=range(max(step,midpoint-radius),min(len(wave)-step,midpoint+radius),step)
    cut=min(candidates,key=lambda x:float(np.mean(wave[x-step//2:x+step//2]**2)))
    words=text.split(); boundary=max(1,min(len(words)-1,round(len(words)*cut/len(wave))))
    return (wave[:cut]," ".join(words[:boundary])),(wave[cut:]," ".join(words[boundary:]))

def verify_modules(model) -> dict[str,int]:
    counts = {"encoder":0,"decoder":0}
    per = {side: {name:0 for name in TARGETS} for side in counts}
    for path, _ in model.named_modules():
        for side in counts:
            if f"model.{side}.layers." in path and path.split(".")[-1] in TARGETS:
                counts[side] += 1; per[side][path.split(".")[-1]] += 1
    for side in counts:
        if any(n == 0 for n in per[side].values()): raise RuntimeError(f"missing LoRA modules in {side}: {per[side]}")
    print("LoRA module counts:",per,flush=True)
    return counts

class AudioRows(Dataset):
    def __init__(self, frame: pd.DataFrame, processor, language, augment: dict):
        self.rows = frame.reset_index(drop=True); self.processor=processor; self.augment=augment
        self.tokenizer = processor.tokenizer
        self.tokenizer.set_prefix_tokens(language=language, task="transcribe", predict_timestamps=False)
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows.iloc[index]; wave=load_audio(ROOT / row.path)
        target = row.text if row.domain == "competition" else to_reference_style(row.text)
        if len(wave)>30*RATE:
            (left_wave,left_text),(right_wave,right_text)=split_long_training(wave,target)
            wave,target=random.choice(((left_wave,left_text),(right_wave,right_text)))
        if self.augment.get("speed_augment"):
            speed=random.choice((1.0,1.1)) if len(wave)>27*RATE else random.choice((0.9,1.0,1.1))
            if speed != 1.0: wave=np.interp(np.arange(0,len(wave),speed), np.arange(len(wave)),wave).astype(np.float32)
        if self.augment.get("gain_augment"):
            wave=np.clip(wave * 10**(random.uniform(-3,3)/20), -1, 1)
        if len(wave) > 30*RATE: raise ValueError("training crop exceeds 30 seconds")
        features=self.processor.feature_extractor(wave,sampling_rate=RATE,return_tensors="pt").input_features[0]
        labels=self.tokenizer(target, return_tensors="pt").input_ids[0][1:]
        if len(labels) > 448: raise ValueError("target exceeds Whisper context")
        return features, labels

def collate(batch):
    features=torch.stack([x[0] for x in batch]); labels=torch.full((len(batch),max(len(x[1]) for x in batch)),-100,dtype=torch.long)
    for i,(_,target) in enumerate(batch): labels[i,:len(target)] = target
    return {"input_features": features,"labels":labels}

def train(config: dict, manifest: Path, output: Path, max_batches: int | None = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    seed=int(config["seed"]); random.seed(seed); np.random.seed(seed);torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
        if hasattr(torch.mps, "set_per_process_memory_fraction"):
            try: torch.mps.set_per_process_memory_fraction(0.9)
            except (RuntimeError, NotImplementedError): pass
    print(f"training device={device}",flush=True)
    frame=pd.read_csv(manifest,sep="\t",keep_default_na=False)
    processor=WhisperProcessor.from_pretrained(config["base_model"],local_files_only=Path(config["base_model"]).exists())
    dtype=torch.bfloat16 if device=="cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model=WhisperForConditionalGeneration.from_pretrained(config["base_model"],torch_dtype=dtype,local_files_only=Path(config["base_model"]).exists())
    model.config.apply_spec_augment=bool(config.get("spec_augment",False))
    model.config.mask_time_prob=0.05 if model.config.apply_spec_augment else 0.0
    model.config.mask_feature_prob=0.05 if model.config.apply_spec_augment else 0.0
    verify_modules(model)
    # Non-reentrant checkpointing retains LoRA gradients when the frozen
    # base receives input features that do not require gradients.
    model.config.use_cache=False; model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    lora=LoraConfig(r=int(config["lora_r"]),lora_alpha=int(config["lora_alpha"]),lora_dropout=float(config["lora_dropout"]),target_modules=list(TARGETS),bias="none")
    model=get_peft_model(model,lora).to(device)
    total=sum(p.numel() for p in model.parameters()); active=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters total={total} trainable={active} percentage={100*active/total:.4f}",flush=True)
    dataset=AudioRows(frame,processor,config.get("language"),config)
    ratio=float(config["competition_weight"])
    weights=[ratio if x=="competition" else 1.0 for x in frame.domain]
    draws=int(sum(x=="jember" for x in frame.domain)+ratio*sum(x=="competition" for x in frame.domain))
    sampler=WeightedRandomSampler(weights,num_samples=draws,replacement=True,generator=torch.Generator().manual_seed(seed))
    loader=DataLoader(dataset,batch_size=int(config["micro_batch_size"]),sampler=sampler,collate_fn=collate,num_workers=0 if device=="mps" else 4,pin_memory=device=="cuda")
    accumulation=math.ceil(int(config["effective_batch_size"])/int(config["micro_batch_size"]))
    batches_per_epoch=min(len(loader),max_batches) if max_batches else len(loader)
    epochs=int(config["epochs"]); total_updates=math.ceil(batches_per_epoch/accumulation)*epochs
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=float(config["learning_rate"]),weight_decay=float(config["weight_decay"]),foreach=False)
    scheduler=get_linear_schedule_with_warmup(optimizer,round(total_updates*float(config["warmup_fraction"])),total_updates)
    output.mkdir(parents=True,exist_ok=True);(output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    (output/"module_counts.json").write_text(json.dumps(verify_modules(model),indent=2)+"\n")
    step=0; model.train(); optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        started=time.monotonic(); loss_sum=0.0
        for batch_index,batch in enumerate(loader):
            if batch_index>=batches_per_epoch: break
            batch={k:v.to(device,non_blocking=device=="cuda") for k,v in batch.items()}
            amp=torch.autocast("cuda",dtype=torch.bfloat16) if device=="cuda" and torch.cuda.is_bf16_supported() else nullcontext()
            with amp:
                loss=model(**batch).loss/accumulation
            loss.backward();loss_sum=loss_sum+loss.detach()*accumulation
            if (batch_index+1)%accumulation==0 or batch_index+1==len(loader):
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),float(config["gradient_clip"]))
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
            if (batch_index+1)%50==0 or batch_index+1==batches_per_epoch:
                print(f"epoch={epoch+1} batch={batch_index+1}/{batches_per_epoch} loss={float(loss.detach()):.4f}",flush=True)
            if batch_index+1 in {math.ceil(batches_per_epoch/2),batches_per_epoch}:
                checkpoint=output/f"epoch_{epoch+1}_{'half' if batch_index+1<batches_per_epoch else 'full'}"
                checkpoint.mkdir(exist_ok=True)
                model.save_pretrained(checkpoint,safe_serialization=True)
                processor.save_pretrained(checkpoint)
                mean_loss=float((loss_sum/(batch_index+1)).detach().cpu())
                (checkpoint/"train_metrics.json").write_text(json.dumps({"epoch":epoch+1,"batch":batch_index+1,"optimizer_step":step,"mean_loss":mean_loss,"elapsed_s":time.monotonic()-started},indent=2)+"\n")
                print("saved",checkpoint,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=HERE/"configs/anchor.json");p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--language",choices=["indonesian","javanese","auto"]);p.add_argument("--ratio",type=int,choices=[2,3,4]);p.add_argument("--seed",type=int);p.add_argument("--base-model");p.add_argument("--max-batches",type=int)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.language: cfg["language"]=None if a.language=="auto" else a.language
    if a.ratio: cfg["competition_weight"]=a.ratio
    if a.seed is not None: cfg["seed"]=a.seed
    if a.base_model: cfg["base_model"]=a.base_model
    train(cfg,a.manifest,a.output,a.max_batches)
