"""Memory-bounded CPU LoRA merge with logits and real-audio token equivalence."""
import argparse
import gc
import json
import os
import sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import pandas as pd
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration,WhisperProcessor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase1_whisper_anchor.inference.audio import load_audio
from overnight_attack.soup import frozen_base_dtype as training_base_dtype

def merge(base,adapter,output,base_dtype=None):
    torch.set_num_threads(4)
    processor=WhisperProcessor.from_pretrained(adapter,local_files_only=True)
    base_dtype=base_dtype or training_base_dtype(adapter)
    # BF16 adaptation quantizes the frozen FP16 checkpoint. Preserve those
    # exact base values before the numerically safer FP32 delta merge.
    model=WhisperForConditionalGeneration.from_pretrained(base,dtype=getattr(torch,base_dtype),local_files_only=True).float()
    model=PeftModel.from_pretrained(model,adapter,local_files_only=True).eval()
    config=model.config
    features=torch.zeros((1,config.num_mel_bins,config.max_source_positions*2))
    ids=torch.tensor([[config.decoder_start_token_id,config.eos_token_id]])
    with torch.inference_mode():before=model(input_features=features,decoder_input_ids=ids).logits.clone()
    manifest=pd.read_csv(ROOT/'phase1_whisper_anchor/data/fold_A_valid.tsv',sep='\t',keep_default_na=False)
    row=manifest.sort_values('duration_s').iloc[0];wave=load_audio(ROOT/row.path)
    inputs=processor.feature_extractor(wave,sampling_rate=16000,return_tensors='pt').input_features
    kwargs={'language':'indonesian','task':'transcribe','num_beams':1,'do_sample':False,'return_timestamps':False,'max_new_tokens':128}
    with torch.inference_mode():tokens_before=model.generate(input_features=inputs,**kwargs).clone()
    merged=model.merge_and_unload(safe_merge=True).eval()
    output.mkdir(parents=True,exist_ok=True)
    merged.save_pretrained(output,safe_serialization=True);processor.save_pretrained(output)
    del model,merged;gc.collect()
    loaded=WhisperForConditionalGeneration.from_pretrained(output,dtype=torch.float32,local_files_only=True).eval()
    with torch.inference_mode():
        after=loaded(input_features=features,decoder_input_ids=ids).logits
        tokens_after=loaded.generate(input_features=inputs,**kwargs)
    difference=float((before-after).abs().max());equal=torch.equal(tokens_before,tokens_after)
    result={'device':'cpu','dtype':'float32','frozen_training_base_dtype':base_dtype,'max_abs_logit_difference':difference,'same_sample_token_ids':equal,
        'equivalent':difference<1e-3 and equal,'output':str(output.resolve()),'note':'Exact adapter/merged comparison in FP32; OOF decode was evaluated separately in its recorded dtype.'}
    (output/'merge_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['equivalent']:raise RuntimeError(result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-model',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--base-dtype',choices=['float32','float16','bfloat16']);a=p.parse_args()
    merge(a.base_model,a.adapter,a.output,a.base_dtype)
