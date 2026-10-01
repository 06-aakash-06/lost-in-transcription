"""Merge decoder LoRA, verify generation equivalence, save local custom code."""
import argparse,json,shutil,sys,gc,time
from pathlib import Path
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from phase2_meralion.inference.meralion import BASE,load,load_audio,transcribe


def merge(adapter,output,base=BASE):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    model,processor,device,dtype=load(base,training=False,adapter=adapter);model.eval()
    load_seconds=time.monotonic()-started
    frame=pd.read_csv(ROOT/'phase1_whisper_anchor/data/fold_A_valid.tsv',sep='\t',keep_default_na=False)
    # Include a short and a naturally >30s clip; never hidden test audio.
    samples=pd.concat([frame.head(1),frame[frame.duration_s>30].head(1)])
    waves=[load_audio(ROOT/p) for p in samples.path]
    started=time.monotonic()
    expected=[transcribe(model,processor,[w],device,dtype)[0] for w in waves]
    adapter_seconds=time.monotonic()-started
    model.text_decoder.model=model.text_decoder.model.merge_and_unload(safe_merge=True)
    started=time.monotonic()
    actual=[transcribe(model,processor,[w],device,dtype)[0] for w in waves]
    result={'samples':len(waves),'device':device,'dtype':str(dtype),'same_transcripts':expected==actual,'adapter':str(adapter),'model_load_seconds':load_seconds,'adapter_inference_seconds':adapter_seconds,'merged_inference_seconds':time.monotonic()-started,'sample_audio_seconds':sum(len(w)/16000 for w in waves)}
    if expected!=actual:raise ValueError('adapter vs merged transcript mismatch; final checkpoint not saved')
    model=model.cpu();gc.collect()
    if device=='mps':torch.mps.empty_cache()
    model.save_pretrained(output,safe_serialization=True,max_shard_size='2GB')
    processor.save_pretrained(output)
    for name in ('configuration_meralion3.py','modeling_meralion3.py','processing_meralion3.py'):
        shutil.copy2(Path(base)/name,output/name)
    for name in ('MERaLiON-3-Public-Licence.pdf','MERaLiON-3-Public-Licence.txt','MODEL_NOTICES.md'):
        src=ROOT/'phase2_meralion'/name
        if src.exists():shutil.copy2(src,output/name)
    (output/'merge_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--base',type=Path,default=BASE);a=p.parse_args();merge(a.adapter,a.output,a.base)
