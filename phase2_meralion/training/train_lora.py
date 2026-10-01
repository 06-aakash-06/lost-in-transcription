"""One decoder-only LoRA recipe. Frozen speech embeddings are cached per clip.

Only supervised answer positions enter the 256k-vocabulary LM head. Its loss
is checkpointed in token chunks, mathematically matching the causal Gemma2
loss without materializing logits for the 300/600 audio prompt tokens.
"""
from __future__ import annotations
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig,get_peft_model
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phase2_meralion.inference.meralion import BASE,load,prompt,move,load_audio,RATE
from phase1_whisper_anchor.training.text import to_reference_style
HERE=ROOT/'phase2_meralion'
TARGETS=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')


def attach_lora(model,cfg):
    model.requires_grad_(False)
    modules=[name for name,mod in model.text_decoder.model.named_modules() if isinstance(mod,torch.nn.Linear) and name.split('.')[-1] in TARGETS]
    counts={target:sum(name.endswith('.'+target) for name in modules) for target in TARGETS}
    if any(value!=26 for value in counts.values()): raise ValueError(f'unexpected Gemma2 projection map: {counts}')
    model.text_decoder.model=get_peft_model(model.text_decoder.model,LoraConfig(r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],target_modules=list(TARGETS),bias='none'))
    model.text_decoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.text_decoder.config.use_cache=False
    model.eval();model.text_decoder.train()
    if any(p.requires_grad for p in model.speech_encoder.parameters()) or any(p.requires_grad for p in model.speech_audio_adapter.parameters()): raise ValueError('speech encoder/projector must remain frozen')
    return counts


def loss_from_hidden(model,hidden,targets,chunk=32):
    cap=model.text_decoder.config.final_logit_softcapping
    def part(h,y):
        logits=model.text_decoder.lm_head(h)
        if cap is not None:logits=torch.tanh(logits/cap)*cap
        return F.cross_entropy(logits.float(),y,reduction='sum')
    flat=hidden.reshape(-1,hidden.shape[-1]);targets=targets.reshape(-1)
    return sum(checkpoint(part,flat[i:i+chunk],targets[i:i+chunk],use_reentrant=False) for i in range(0,len(targets),chunk))/len(targets)


def supervised_loss(model,cached,answer_ids,device,dtype,chunk):
    prefix=cached['input_ids'].to(device)
    answer=torch.tensor([answer_ids],device=device,dtype=torch.long)
    ids=torch.cat([prefix,answer],dim=1)
    with torch.no_grad():
        embeds=model.text_decoder.get_input_embeddings()(ids)
        mask=(ids==model.config.speech_token_index).unsqueeze(-1).expand_as(embeds)
        embeds=embeds.masked_scatter(mask,cached['speech'].to(device,dtype=dtype))
    hidden=model.text_decoder.model(inputs_embeds=embeds,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
    return loss_from_hidden(model,hidden[:,prefix.shape[1]-1:-1,:],answer,chunk)


def safe_backward(model,cached,answer_ids,params,device,dtype,chunk,accumulation):
    # Preserve the already accumulated gradients if a single MPS operation
    # produces nonfinite values. Recompute that example, without dropping it.
    previous=[p.grad.detach().clone() if p.grad is not None else None for p in params]
    loss=supervised_loss(model,cached,answer_ids,device,dtype,chunk)
    finite=bool(torch.isfinite(loss))
    if finite:
        (loss/accumulation).backward()
        finite=bool(torch.stack([torch.isfinite(p.grad).all() for p in params if p.grad is not None]).all())
    if finite:return float(loss.detach().cpu()),False
    if device!='mps':raise FloatingPointError('nonfinite loss or gradient')
    del loss
    for p,g in zip(params,previous):p.grad=g
    buffers={name:buf.detach().cpu().clone() for name,buf in model.text_decoder.named_buffers()}
    extra_precision=[(p,p.detach().cpu().clone()) for p in model.text_decoder.parameters() if not p.requires_grad and p.dtype!=dtype]
    # Only the decoder involved in this example falls back. No optimizer step
    # occurs on CPU, and the frozen speech encoder remains on MPS.
    model.text_decoder.to(device='cpu',dtype=torch.float32)
    torch.mps.empty_cache()
    cpu_loss=supervised_loss(model,cached,answer_ids,'cpu',torch.float32,chunk)
    if not torch.isfinite(cpu_loss):raise FloatingPointError('CPU FP32 fallback loss is nonfinite')
    (cpu_loss/accumulation).backward()
    if not bool(torch.stack([torch.isfinite(p.grad).all() for p in params if p.grad is not None]).all()):
        raise FloatingPointError('CPU FP32 fallback gradient is nonfinite')
    value=float(cpu_loss.detach());del cpu_loss
    adapter_state=[(p.detach().clone(),p.grad.detach().clone() if p.grad is not None else None) for p in params]
    model.text_decoder.to(device=device,dtype=dtype)
    # PEFT optimizes its adapter weights in FP32 even with a BF16 base.
    # Restore the CPU FP32 values directly, avoiding a BF16 round trip.
    for p,(data,grad) in zip(params,adapter_state):
        p.data=data.to(device)
        p.grad=grad.to(device) if grad is not None else None
    for name,buf in model.text_decoder.named_buffers():buf.data=buffers[name].to(device)
    for p,data in extra_precision:p.data=data.to(device)
    return value,True


class CachedSpeech:
    def __init__(self,base,model,processor,device,dtype):
        self.model,self.processor,self.device,self.dtype=model,processor,device,dtype
        fingerprint=hashlib.sha256((Path(base)/'config.json').read_bytes()+str([(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(Path(base).glob('*.safetensors'))]).encode()).hexdigest()[:16]
        self.path=HERE/'cache'/fingerprint;self.path.mkdir(parents=True,exist_ok=True)
    def get(self,row):
        path=self.path/(row.clip_id+'.pt')
        if path.exists():return torch.load(path,map_location='cpu',weights_only=True)
        inputs=self.processor(text=[prompt(self.processor)],audios=[load_audio(ROOT/row.path)],sampling_rate=RATE,return_tensors='pt',padding=True)
        moved=move(inputs,self.device,self.dtype)
        with torch.no_grad():
            speech=self.model.speech_encoder(moved['input_features'],attention_mask=moved['feature_attention_mask']).last_hidden_state
            speech=self.model.speech_audio_adapter(self.model.ln_speech(speech))
        data={'input_ids':inputs['input_ids'].cpu(),'speech':speech.cpu()}
        temporary=path.with_suffix('.tmp');torch.save(data,temporary);os.replace(temporary,path)
        return data


def train(manifest,output,base=BASE,max_batches=None,stop_after_half=False):
    cfg=json.loads((HERE/'configs/decoder_lora.json').read_text());seed=cfg['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model,processor,device,dtype=load(base,training=True)
    modules=attach_lora(model,cfg)
    if device=='mps':torch.mps.manual_seed(seed)
    total=sum(p.numel() for p in model.parameters());active=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'device={device} dtype={dtype} total={total} trainable={active} percent={100*active/total:.4f} modules={modules}',flush=True)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],foreach=False)
    frame=pd.read_csv(manifest,sep='\t',keep_default_na=False)
    draws=int(sum(frame.domain=='jember')+cfg['competition_weight']*sum(frame.domain=='competition'))
    if max_batches:draws=min(draws,max_batches)
    weights=[cfg['competition_weight'] if domain=='competition' else 1 for domain in frame.domain]
    sampler=WeightedRandomSampler(weights,draws,replacement=True,generator=torch.Generator().manual_seed(seed))
    accumulation=cfg['gradient_accumulation'];updates=math.ceil(draws/accumulation)
    scheduler=get_linear_schedule_with_warmup(optimizer,round(updates*cfg['warmup_fraction']),updates)
    cache=CachedSpeech(base,model,processor,device,dtype)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cfg.update({'base_model':str(base),'manifest':str(manifest),'device':device,'dtype':str(dtype),'projection_counts':modules,'total_parameters':total,'trainable_parameters':active,'draws':draws})
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    started=time.monotonic();loss_sum=0.;steps=0;fallbacks=0;optimizer.zero_grad(set_to_none=True)
    half=min(draws,math.ceil(updates/2)*accumulation)
    for index,row_index in enumerate(sampler):
        row=frame.iloc[row_index];cached=cache.get(row)
        target=row.text if row.domain=='competition' else to_reference_style(row.text)
        answer=processor.tokenizer(target,add_special_tokens=False).input_ids+[processor.tokenizer.eos_token_id]
        loss,fallback=safe_backward(model,cached,answer,params,device,dtype,cfg['loss_chunk_tokens'],accumulation)
        loss_sum+=loss
        fallbacks+=int(fallback)
        if fallback:print(f'CPU FP32 decoder fallback at batch={index+1}; example retained',flush=True)
        if (index+1)%accumulation==0 or index+1==draws:
            if index+1==draws and draws%accumulation:
                for param in params:
                    if param.grad is not None:param.grad.mul_(accumulation/(draws%accumulation))
            torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);steps+=1
        if (index+1)%10==0 or index+1==draws:
            elapsed=time.monotonic()-started
            print(f'batch={index+1}/{draws} loss={loss:.4f} elapsed={elapsed:.1f} projected_epoch_s={elapsed/(index+1)*draws:.0f}',flush=True)
        if index+1 in (half,draws):
            name='epoch_1_half' if index+1<draws else 'epoch_1_full'
            destination=output/name;destination.mkdir(exist_ok=True)
            model.text_decoder.model.save_pretrained(destination,safe_serialization=True,save_embedding_layers=False)
            processor.save_pretrained(destination)
            (destination/'train_metrics.json').write_text(json.dumps({'batches':index+1,'optimizer_steps':steps,'mean_loss':loss_sum/(index+1),'elapsed_s':time.monotonic()-started,'cpu_decoder_retries':fallbacks},indent=2)+'\n')
            print('saved',destination,flush=True)
            if stop_after_half and index+1==half:break
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--base',type=Path,default=BASE);p.add_argument('--max-batches',type=int);p.add_argument('--stop-after-half',action='store_true');a=p.parse_args();train(a.manifest,a.output,a.base,a.max_batches,a.stop_after_half)
