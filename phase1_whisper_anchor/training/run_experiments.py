"""Launch conversation-disjoint language/ratio grid and compile selection table."""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
DEFAULT_BASE=ROOT/"submission_src/models/whisper_large_v3_turbo"

def call(*parts): subprocess.run([sys.executable,*map(str,parts)],check=True)

def run(base_model: str, languages: list[str], ratios: list[int], output: Path):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for language in languages:
        for ratio in ratios:
            for fold in ("fold_A","fold_B"):
                run_dir=output/f"{language}_x{ratio}_{fold}"
                if not any(run_dir.glob("epoch_*_full/adapter_config.json")):
                    call(HERE/"training/train_whisper_lora.py","--manifest",HERE/f"data/{fold}_train.tsv","--output",run_dir,"--language",language,"--ratio",ratio,"--base-model",base_model)
                for checkpoint in sorted(run_dir.glob("epoch_*")):
                    if not (checkpoint/"adapter_config.json").exists(): continue
                    metric_path=checkpoint/"valid.json"
                    if not metric_path.exists():
                        call(HERE/"training/evaluate.py","--manifest",HERE/f"data/{fold}_valid.tsv","--checkpoint",checkpoint,"--output",metric_path,"--base-model",base_model,"--language",language)
                    held_path=checkpoint/"jember.json"
                    if not held_path.exists():
                        call(HERE/"training/evaluate.py","--manifest",HERE/"data/jember_holdout.tsv","--checkpoint",checkpoint,"--output",held_path,"--base-model",base_model,"--language",language)
    for language in languages:
        for ratio in ratios:
            a_dir=output/f"{language}_x{ratio}_fold_A";b_dir=output/f"{language}_x{ratio}_fold_B"
            for name in sorted({p.name for p in a_dir.glob("epoch_*")} & {p.name for p in b_dir.glob("epoch_*")}):
                a=json.loads((a_dir/name/"valid.json").read_text());b=json.loads((b_dir/name/"valid.json").read_text());h1=json.loads((a_dir/name/"jember.json").read_text());h2=json.loads((b_dir/name/"jember.json").read_text())
                errors=sum(a[k]+b[k] for k in ("substitutions","deletions","insertions"));words=a["words"]+b["words"]
                rows.append({"configuration":f"{language}_x{ratio}_{name}","language":language,"ratio":ratio,"checkpoint":name,"fold_A_WER":a["wer"],"fold_B_WER":b["wer"],"worst_fold_WER":max(a["wer"],b["wer"]),"aggregate_WER":errors/words,"Jember_holdout_WER":(h1["wer"]+h2["wer"])/2,"substitutions":a["substitutions"]+b["substitutions"],"deletions":a["deletions"]+b["deletions"],"insertions":a["insertions"]+b["insertions"],"runtime_s":a["elapsed_s"]+b["elapsed_s"]})
    existing=HERE/"results/language_token.csv"
    if existing.exists():
        with existing.open(newline="") as f:
            for old in csv.DictReader(f):
                if old.get("configuration") and old["configuration"] not in {r["configuration"] for r in rows}: rows.append(old)
    rows.sort(key=lambda x:(float(x["worst_fold_WER"]),float(x["aggregate_WER"]),float(x["Jember_holdout_WER"])))
    if rows:
        for path in (HERE/"results/language_token.csv",output/"selection.csv"):
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print("selected",rows[0])

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--base-model",default=str(DEFAULT_BASE));p.add_argument("--languages",nargs="+",default=["indonesian","javanese","auto"]);p.add_argument("--ratios",nargs="+",type=int,default=[3]);p.add_argument("--output",type=Path,default=HERE/"runs");a=p.parse_args();run(a.base_model,a.languages,a.ratios,a.output)
