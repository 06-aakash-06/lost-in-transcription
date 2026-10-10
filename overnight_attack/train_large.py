"""Measured MPS Large-v3 LoRA; periodic resumable and graceful-stop saves."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import re
import signal
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration,WhisperProcessor,get_linear_schedule_with_warmup
from peft import LoraConfig,get_peft_model,load_peft_weights,set_peft_model_state_dict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'phase1_whisper_anchor'))
from training.train_whisper_lora import AudioRows,collate

def train(config,manifest,output,resume=None,max_batches=None,max_runtime=None,stop_after=None):
    if output.exists() and any(output.iterdir()) and resume is None:raise FileExistsError(output)
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device=='cpu':raise RuntimeError('No usable training accelerator')
    seed=int(config['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=='mps':torch.mps.manual_seed(seed);torch.mps.set_per_process_memory_fraction(.88)
    else:torch.cuda.manual_seed_all(seed)
    dtype=getattr(torch,config['dtype'])
    frame=pd.read_csv(manifest,sep='\t',keep_default_na=False)
    hold=pd.read_csv(ROOT/'phase1_whisper_anchor/data/jember_holdout.tsv',sep='\t',keep_default_na=False)
    assert frame.clip_id.is_unique and not set(frame.clip_id)&set(hold.clip_id)
    assert set(frame[frame.domain=='jember'].session).isdisjoint(set(hold.session))
    valid_name=manifest.name.replace('_train.tsv','_valid.tsv')
    if valid_name!=manifest.name and (manifest.parent/valid_name).exists():
        valid=pd.read_csv(manifest.parent/valid_name,sep='\t',keep_default_na=False)
        assert not set(frame.clip_id)&set(valid.clip_id)
        assert set(frame[frame.domain=='competition'].convo_id).isdisjoint(set(valid.convo_id))
    processor=WhisperProcessor.from_pretrained(config['base_model'],local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(config['base_model'],dtype=dtype,local_files_only=True,attn_implementation='sdpa')
    assert model.config.decoder_layers==32,'Requested Large-v3, not Turbo'
    model.config.apply_spec_augment=bool(config.get('spec_augment'))
    model.config.mask_time_prob=.05 if model.config.apply_spec_augment else 0.
    model.config.mask_feature_prob=.05 if model.config.apply_spec_augment else 0.
    model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    projections=r'(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|encoder_attn\.(?:q_proj|k_proj|v_proj|out_proj)|fc1|fc2)'
    target=r'.*\.decoder\.layers\.\d+\.'+projections
    count=int(config['encoder_lora_last_n'])
    if count:
        layers='|'.join(map(str,range(model.config.encoder_layers-count,model.config.encoder_layers)))
        target='(?:'+target+r'|.*\.encoder\.layers\.(?:'+layers+r')\.'+projections+')'
    model=get_peft_model(model,LoraConfig(r=config['lora_r'],lora_alpha=config['lora_alpha'],lora_dropout=config['lora_dropout'],target_modules=target,bias='none')).to(device)
    trainable={name:p for name,p in model.named_parameters() if p.requires_grad}
    active=sum(p.numel() for p in trainable.values())
    assert any('decoder.layers.31' in n for n in trainable)
    print(f'device={device} dtype={dtype} trainable={active} encoder_last_n={count} decoder_layers=32',flush=True)
    weights=torch.tensor([float(config['competition_weight']) if d=='competition' else 1. for d in frame.domain],dtype=torch.double)
    draws=int(weights.sum());micro=int(config['micro_batch_size']);batches=math.ceil(draws/micro)
    if max_batches:batches=min(batches,max_batches);draws=batches*micro
    accumulation=math.ceil(config['effective_batch_size']/micro)
    total_updates=math.ceil(batches/accumulation)*int(config['epochs'])
    optimizer=torch.optim.AdamW(trainable.values(),lr=config['learning_rate'],weight_decay=config['weight_decay'],foreach=False)
    scheduler=get_linear_schedule_with_warmup(optimizer,round(total_updates*config['warmup_fraction']),total_updates)
    generator=torch.Generator().manual_seed(seed);state=None
    if resume:
        state=torch.load(resume/'training_state.pt',map_location='cpu',weights_only=False)
        assert state['config']==config and state['manifest']==str(manifest.resolve())
        set_peft_model_state_dict(model,load_peft_weights(str(resume),device='cpu'))
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);generator.set_state(state['sampler_rng'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'provenance.json').write_text(json.dumps({'manifest':str(manifest.resolve()),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'domain_rows':frame.domain.value_counts().to_dict(),'draws':draws,'trainable_parameters':active,'target_modules':target},indent=2)+'\n')
    started_run=time.monotonic();prior_elapsed=state.get('total_elapsed_s',0.) if state else 0.
    step=state['step'] if state else 0;start_epoch=state['epoch'] if state else 0
    model.train();optimizer.zero_grad(set_to_none=True)
    if state:
        for name,gradient in state.get('gradients',{}).items():trainable[name].grad=gradient.to(device=device,dtype=trainable[name].dtype)
    stopped={'requested':False}
    def request_stop(signum,frame):stopped['requested']=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    first_name=next(n for n in trainable if '.lora_B.' in n)
    original=trainable[first_name].detach().cpu().clone()
    for epoch in range(start_epoch,int(config['epochs'])):
        continuing=state is not None and epoch==state['epoch'] and state['indices'] is not None
        indices=state['indices'] if continuing else torch.multinomial(weights,draws,replacement=True,generator=generator).tolist()
        offset=state['next_batch'] if continuing else 0;loss_sum=state['loss_sum'] if continuing else 0.
        epoch_prior=state['epoch_elapsed_s'] if continuing else 0.
        dataset=AudioRows(frame,processor,config['language'],config)
        loader=DataLoader(dataset,batch_size=micro,sampler=indices[offset*micro:],collate_fn=collate,num_workers=0)
        iterator=iter(loader)
        if state is not None and epoch==state['epoch']:
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
            if device=='mps':torch.mps.set_rng_state(state['device_rng'])
            else:torch.cuda.set_rng_state_all(state['device_rng'])
        started=time.monotonic();half=min(batches,math.ceil(math.ceil(batches/2)/accumulation)*accumulation)
        for batch_index,batch in enumerate(iterator,offset):
            batch={k:v.to(device=device,dtype=dtype if k=='input_features' else v.dtype) for k,v in batch.items()}
            loss=model(**batch).loss;value=float(loss.detach().float())
            if not math.isfinite(value):raise FloatingPointError(f'nonfinite loss batch={batch_index+1}')
            (loss/accumulation).backward();loss_sum+=value
            completed=batch_index+1
            if completed%accumulation==0 or completed==batches:
                norm=torch.nn.utils.clip_grad_norm_(trainable.values(),config['gradient_clip'],error_if_nonfinite=True)
                if not math.isfinite(float(norm)):raise FloatingPointError('nonfinite gradient')
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
            elapsed=time.monotonic()-started+epoch_prior
            if completed%10==0 or completed==batches:
                stats={'epoch':epoch+1,'batch':completed,'batches_per_epoch':batches,'optimizer_step':step,'mean_loss':loss_sum/completed,
                    'last_loss':value,'learning_rate':scheduler.get_last_lr()[0],'elapsed_s':elapsed,'total_elapsed_s':time.monotonic()-started_run+prior_elapsed,
                    'seconds_per_new_batch':(time.monotonic()-started)/max(1,completed-offset),
                    'mps_driver_gb':torch.mps.driver_allocated_memory()/1e9 if device=='mps' else None}
                (output/'progress.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats),flush=True)
            if max_runtime and time.monotonic()-started_run>=max_runtime:stopped['requested']=True
            if stop_after=='half' and completed==half:stopped['requested']=True
            periodic=completed%int(config.get('save_every_batches',256))==0
            if completed in {half,batches} or periodic or stopped['requested']:
                name=f"epoch_{epoch+1}_{'half' if completed==half and half!=batches else 'full' if completed==batches else 'batch_'+str(completed).zfill(6)}"
                checkpoint=output/name
                if checkpoint.exists():raise FileExistsError(checkpoint)
                checkpoint.mkdir();model.save_pretrained(checkpoint,safe_serialization=True);processor.save_pretrained(checkpoint)
                saved={'config':config,'manifest':str(manifest.resolve()),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'epoch':epoch,'next_batch':completed,'indices':indices,'sampler_rng':generator.get_state(),'python_rng':random.getstate(),
                    'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'device_rng':torch.mps.get_rng_state() if device=='mps' else torch.cuda.get_rng_state_all(),
                    'step':step,'loss_sum':loss_sum,'epoch_elapsed_s':elapsed,'total_elapsed_s':time.monotonic()-started_run+prior_elapsed,
                    'gradients':{n:p.grad.detach().cpu().clone() for n,p in trainable.items() if p.grad is not None},'accumulated_microbatches':completed%accumulation}
                if completed==batches:saved.update(epoch=epoch+1,next_batch=0,indices=None,loss_sum=0.,epoch_elapsed_s=0.,gradients={},accumulated_microbatches=0)
                temporary=checkpoint/'training_state.tmp';torch.save(saved,temporary);temporary.replace(checkpoint/'training_state.pt')
                metrics={'epoch':epoch+1,'batch':completed,'epoch_fraction':epoch+completed/batches,'optimizer_step':step,'mean_loss':loss_sum/completed,
                    'elapsed_s':elapsed,'total_elapsed_s':saved['total_elapsed_s'],'lora_weight_change_max':float((trainable[first_name].detach().cpu()-original).abs().max())}
                (checkpoint/'train_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
                (output/'latest_checkpoint.txt').write_text(str(checkpoint.resolve())+'\n')
                print('SAVED',checkpoint,json.dumps(metrics),flush=True)
                if stopped['requested']:
                    print('GRACEFUL STOP',checkpoint,flush=True);return
        state=None
    (output/'completed.json').write_text(json.dumps({'completed':True,'optimizer_steps':step,'elapsed_s':time.monotonic()-started_run+prior_elapsed},indent=2)+'\n')
    print('TRAINING COMPLETE',output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--max-batches',type=int);p.add_argument('--max-runtime',type=float)
    p.add_argument('--stop-after',choices=['half'])
    a=p.parse_args();train(json.loads(a.config.read_text()),a.manifest,a.output,a.resume,a.max_batches,a.max_runtime,a.stop_after)
