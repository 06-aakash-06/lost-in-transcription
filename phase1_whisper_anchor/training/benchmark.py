"""Benchmark a merged model on dev audio; never use hidden-test statistics."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import pandas as pd
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from inference.audio import load_audio, RATE
from inference.whisper import WhisperAnchor

def benchmark(model: Path, manifest: Path, output: Path, language: str|None, batch_size: int, count: int=100, long_mode: str="native") -> dict:
    frame=pd.read_csv(manifest,sep="\t").head(count)
    device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device=="cuda": torch.cuda.reset_peak_memory_stats()
    begin=time.monotonic();anchor=WhisperAnchor(model,language,batch_size,long_mode);load_s=time.monotonic()-begin
    waves=[load_audio(HERE.parent/p) for p in frame.path]
    start=time.monotonic()
    for i in range(0,len(waves),batch_size): anchor.transcribe_arrays(waves[i:i+batch_size])
    if device=="cuda": torch.cuda.synchronize()
    elif device=="mps": torch.mps.synchronize()
    duration=time.monotonic()-start
    result={"model_load_s":load_s,"inference_s":duration,"clips":len(waves),"batch_size":batch_size,"clips_per_s":len(waves)/duration,"audio_s_per_s":sum(map(len,waves))/RATE/duration,"peak_memory_gb":torch.cuda.max_memory_allocated()/1e9 if device=="cuda" else None,"projected_2118_s":load_s+2118*duration/len(waves),"device":anchor.device}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+"\n");return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",type=Path,required=True);p.add_argument("--manifest",type=Path,default=HERE/"data/fold_A_valid.tsv");p.add_argument("--output",type=Path,default=HERE/"results/benchmark.json");p.add_argument("--language",choices=["indonesian","javanese","auto"],default="indonesian");p.add_argument("--batch-size",type=int,default=8);p.add_argument("--count",type=int,default=100);p.add_argument("--long-mode",choices=["overlap","native"],default="native");a=p.parse_args();print(benchmark(a.model,a.manifest,a.output,None if a.language=="auto" else a.language,a.batch_size,a.count,a.long_mode))
