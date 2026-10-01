"""Train all dev clips using selected fold recipe, then merge and package."""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
DEFAULT_BASE=ROOT/"submission_src/models/whisper_large_v3_turbo"

def command(*args): subprocess.run([sys.executable,*map(str,args)],check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--selection",type=Path,default=HERE/"results/language_token.csv");parser.add_argument("--base-model",default=str(DEFAULT_BASE));parser.add_argument("--output",type=Path,default=HERE/"models/final_merged");a=parser.parse_args()
    with a.selection.open(newline="") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError("no evaluated fold results; final training would be unselected")
    best=min(rows,key=lambda r:(float(r["worst_fold_WER"]),float(r["aggregate_WER"]),float(r["Jember_holdout_WER"])))
    language=best["language"];ratio=int(best["ratio"]);checkpoint=best["checkpoint"]
    run=HERE/"runs"/f"final_{language}_x{ratio}"
    if not (run/checkpoint/"adapter_config.json").exists():
        command(HERE/"training/train_whisper_lora.py","--manifest",HERE/"data/final_train.tsv","--output",run,"--language",language,"--ratio",ratio,"--base-model",a.base_model)
    command(HERE/"training/merge_lora.py","--base-model",a.base_model,"--adapter",run/checkpoint,"--output",a.output,"--language",language)
    command(HERE/"submission/build_submission.py","--model",a.output,"--language",language)
    print("selected",best["configuration"])

if __name__=="__main__": main()
