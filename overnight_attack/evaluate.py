"""Restartable native-long-form Whisper decoding for Large-v3 and Turbo."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import pandas as pd
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration,WhisperProcessor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase3_protected_fusion.runtime.whisper import WhisperAnchor
from phase1_whisper_anchor.inference.audio import load_audio,RATE
from phase3_protected_fusion.runtime.postprocess import clean
from phase3_protected_fusion.runtime.fusion import collapse_single_runs
from phase1_whisper_anchor.training.metrics import corpus

def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

class Engine(WhisperAnchor):
    def __init__(self,base,adapter,dtype='float32',language='indonesian',beams=1,batch_size=4):
        self.device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.dtype=getattr(torch,dtype);self.language=None if language=='auto' else language
        self.batch_size=batch_size;self.long_mode='native';self.beams=beams
        if self.device=='mps':torch.mps.set_per_process_memory_fraction(.88)
        self.processor=WhisperProcessor.from_pretrained(adapter or base,local_files_only=True)
        model=WhisperForConditionalGeneration.from_pretrained(base,dtype=self.dtype,local_files_only=True,attn_implementation='sdpa')
        if adapter:model=PeftModel.from_pretrained(model,adapter,local_files_only=True)
        self.model=model.to(self.device).eval()
        self.model.generation_config.language=None

    @torch.inference_mode()
    def _decode(self,waves,long_form=False):
        if long_form:
            assert len(waves)==1
            features=self.processor.feature_extractor(waves[0],sampling_rate=RATE,truncation=False,padding='longest',return_tensors='pt',return_attention_mask=True)
        else:features=self.processor.feature_extractor(waves,sampling_rate=RATE,return_tensors='pt',return_attention_mask=True)
        kwargs={'num_beams':self.beams,'do_sample':False,'task':'transcribe','return_timestamps':long_form,'max_new_tokens':440}
        if self.language is not None:kwargs['language']=self.language
        ids=self.model.generate(input_features=features.input_features.to(self.device,dtype=self.dtype),attention_mask=features.attention_mask.to(self.device),**kwargs)
        return [clean(s) for s in self.processor.batch_decode(ids,skip_special_tokens=True)]

def evaluate(manifest,checkpoint,output,base,dtype='float32',language='indonesian',beams=1,batch_size=4):
    frame=pd.read_csv(manifest,sep='\t',keep_default_na=False);assert frame.clip_id.is_unique
    weight=checkpoint/'adapter_model.safetensors' if checkpoint else base/'model.safetensors'
    with weight.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    provenance={'checkpoint':str(checkpoint.resolve()) if checkpoint else None,'base':str(base.resolve()),'weight_sha256':digest,
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'language':language,'beams':beams,'dtype':dtype,'batch_size':batch_size,'long_mode':'native'}
    output.parent.mkdir(parents=True,exist_ok=True);cache_path=output.with_suffix('.cache.json')
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {'provenance':provenance,'predictions':{},'elapsed_s':0.}
    assert cache['provenance']==provenance,'cached prediction provenance mismatch'
    cache_pred=cache['predictions'];assert set(cache_pred)<=set(frame.clip_id)
    for row in frame.itertuples():
        if row.clip_id in cache_pred:assert cache_pred[row.clip_id]['reference']==row.text
    remaining=frame[~frame.clip_id.isin(cache_pred)]
    if len(remaining):
        torch.manual_seed(1337);engine=Engine(base,checkpoint,dtype,language,beams,batch_size)
        started=time.monotonic();previous=cache['elapsed_s']
        for start in range(0,len(remaining),batch_size):
            part=remaining.iloc[start:start+batch_size];waves=[load_audio(ROOT/p) for p in part.path]
            texts=engine.transcribe_arrays(waves);assert len(texts)==len(part)
            for row,text in zip(part.itertuples(),texts):cache_pred[row.clip_id]={'reference':row.text,'transcript':text}
            cache['elapsed_s']=previous+time.monotonic()-started;write(cache_path,cache)
            if (start//batch_size)%5==0:print(f'{output.stem} clips={len(cache_pred)}/{len(frame)} elapsed_s={cache["elapsed_s"]:.1f}',flush=True)
        del engine
    texts=[cache_pred[c]['transcript'] for c in frame.clip_id]
    result={'provenance':provenance,'clips':len(frame),'raw':corpus(frame.text.tolist(),texts),
        'safe':corpus(frame.text.tolist(),list(map(collapse_single_runs,texts))),'elapsed_s':cache['elapsed_s']}
    pd.DataFrame({'clip_id':frame.clip_id,'reference':frame.text,'transcript':texts}).to_csv(output.with_suffix('.csv'),index=False)
    write(output,result);print('EVALUATION COMPLETE',output,json.dumps({'raw':result['raw']['wer'],'safe':result['safe']['wer']}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--dtype',choices=['float32','bfloat16','float16'],default='float32');p.add_argument('--language',choices=['indonesian','auto','javanese'],default='indonesian')
    p.add_argument('--beams',type=int,default=1);p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args();evaluate(a.manifest,a.checkpoint,a.output,a.base,a.dtype,a.language,a.beams,a.batch_size)
