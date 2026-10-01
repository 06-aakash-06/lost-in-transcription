"""Run the requested MPS language sweep, ratio sweep, final merge and ZIP smoke."""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent

def run(*args): subprocess.run([sys.executable,*map(str,args)],check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--base-model",default=str(ROOT/"submission_src/models/whisper_large_v3_turbo"));p.add_argument("--skip-local-zip-test",action="store_true");a=p.parse_args()
    experiments=HERE/"training/run_experiments.py"
    run(experiments,"--base-model",a.base_model,"--languages","indonesian","javanese","auto","--ratios","3")
    with (HERE/"results/language_token.csv").open(newline="") as f: rows=[r for r in csv.DictReader(f) if int(r["ratio"])==3]
    if not rows: raise RuntimeError("language-token sweep produced no fold metrics")
    best=min(rows,key=lambda r:(float(r["worst_fold_WER"]),float(r["aggregate_WER"]),float(r["Jember_holdout_WER"])))
    language=best["language"]
    print(f"ratio-3 selected language={language} checkpoint={best['checkpoint']}",flush=True)
    run(experiments,"--base-model",a.base_model,"--languages",language,"--ratios","2","4")
    run(HERE/"training/finalize.py","--base-model",a.base_model)
    if not a.skip_local_zip_test:
        run(HERE/"submission/test_zip_local.py","--zip",ROOT/"artifacts/phase1_whisper_anchor_submission.zip","--count","2")

if __name__=="__main__": main()
