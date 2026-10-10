"""Audit the precise merged-then-cast inference representation to be shipped."""
import os,sys,json,argparse,gc,hashlib
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration,WhisperProcessor
from overnight_attack import evaluate as original
from overnight_attack.soup import frozen_base_dtype

class MergedEngine(original.Engine):
    def __init__(self,base,adapter,dtype='bfloat16',language='indonesian',beams=1,batch_size=4):
        self.device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.dtype=getattr(torch,dtype);self.language=None if language=='auto' else language
        self.batch_size=batch_size;self.long_mode='native';self.beams=beams
        torch.set_num_threads(4)
        if self.device=='mps':torch.mps.set_per_process_memory_fraction(.88)
        self.processor=WhisperProcessor.from_pretrained(adapter,local_files_only=True)
        model=WhisperForConditionalGeneration.from_pretrained(base,dtype=getattr(torch,frozen_base_dtype(adapter)),local_files_only=True,attn_implementation='sdpa').float()
        model=PeftModel.from_pretrained(model,adapter,local_files_only=True).eval()
        self.model=model.merge_and_unload(safe_merge=True).to(device=self.device,dtype=self.dtype).eval()
        self.model.generation_config.language=None
        gc.collect()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dtype',default='bfloat16');p.add_argument('--language',default='indonesian');p.add_argument('--beams',type=int,default=1);p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    mode={'mode':'frozen training base cast to FP32; LoRA delta safe merged in FP32; merged weights cast to inference dtype','dtype':a.dtype,'adapter':str(a.checkpoint.resolve()),'frozen_base_dtype':frozen_base_dtype(a.checkpoint)}
    path=a.output.with_suffix('.mode.json')
    if path.exists():assert json.loads(path.read_text())==mode
    else:path.write_text(json.dumps(mode,indent=2)+'\n')
    original.Engine=MergedEngine
    original.evaluate(a.manifest,a.checkpoint,a.output,ROOT/'overnight_attack/models/large_v3_base',a.dtype,a.language,a.beams,a.batch_size)
