"""Evaluate one checkpoint on a saved manifest with official WER and S/D/I."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from inference.audio import load_audio
from inference.whisper import WhisperAnchor
from inference.postprocess import catastrophic_repeat, collapse_catastrophic
from training.metrics import corpus

def evaluate(manifest: Path, checkpoint: Path, output: Path, base_model: str|None, language: str|None, batch_size: int=8, long_mode: str="native") -> dict:
    frame=pd.read_csv(manifest,sep="\t",keep_default_na=False)
    anchor=WhisperAnchor.from_adapter(base_model,checkpoint,language,batch_size,long_mode) if base_model else WhisperAnchor(checkpoint,language,batch_size,long_mode)
    started=time.monotonic();texts=[]
    for start in range(0,len(frame),batch_size):
        waves=[load_audio(HERE.parent/path) for path in frame.path.iloc[start:start+batch_size]]
        texts.extend(anchor.transcribe_arrays(waves))
    elapsed=time.monotonic()-started
    stats=corpus(frame.text.tolist(),texts)
    stats["repeat_collapse_wer"] = corpus(frame.text.tolist(),[collapse_catastrophic(x) for x in texts])["wer"]
    stats.update({"checkpoint":str(checkpoint),"manifest":str(manifest),"language":language,"batch_size":batch_size,"long_mode":long_mode,"elapsed_s":elapsed,"clips_per_s":len(frame)/elapsed,"catastrophic_repeats":sum(map(catastrophic_repeat,texts))})
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"clip_id":frame.clip_id,"reference":frame.text,"transcript":texts}).to_csv(output.with_suffix(".csv"),index=False)
    output.write_text(json.dumps(stats,indent=2)+"\n")
    return stats

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--base-model");p.add_argument("--language",choices=["indonesian","javanese","auto"],default="indonesian");p.add_argument("--batch-size",type=int,default=8);p.add_argument("--long-mode",choices=["overlap","native"],default="native")
    a=p.parse_args();print(evaluate(a.manifest,a.checkpoint,a.output,a.base_model,None if a.language=="auto" else a.language,a.batch_size,a.long_mode))
