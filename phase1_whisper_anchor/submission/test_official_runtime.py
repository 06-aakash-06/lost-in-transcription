"""Test the exact final ZIP in the official network-disabled Docker image."""
from __future__ import annotations
import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def run(runtime: Path, archive: Path):
    runtime=runtime.resolve();archive=archive.resolve()
    if not (runtime/"score.py").is_file(): raise FileNotFoundError("official runtime checkout missing")
    expected=archive.with_suffix(archive.suffix+".sha256").read_text().split()[0]
    if sha(archive)!=expected: raise ValueError("submission ZIP SHA256 differs from builder sidecar")
    data=runtime/"data/data";clips=data/"clips";submission=runtime/"submission"
    clips.mkdir(parents=True,exist_ok=True);submission.mkdir(parents=True,exist_ok=True)
    dev=pd.read_csv(ROOT/"indonesian_dev/metadata.tsv",sep="\t",keep_default_na=False)
    for name in dev.audio_filename: shutil.copy2(ROOT/"indonesian_dev/clips"/name,clips/name)
    dev.drop(columns=["transcript"]).to_csv(data/"test_metadata.csv",index=False)
    reference=runtime/"data/dev_reference.csv"
    dev[["audio_filename","transcript"]].to_csv(reference,index=False)
    target=submission/"submission.zip";shutil.copy2(archive,target)
    if sha(target)!=expected: raise ValueError("copied ZIP differs from source")
    image="lostintranscriptionprodacr.azurecr.io/lost-in-transcription-competition:latest"
    subprocess.run(["docker","run","--rm","--gpus","all","--network","none","--mount",f"type=bind,source={data},target=/code_execution/data,readonly","--mount",f"type=bind,source={submission},target=/code_execution/submission",image],check=True)
    result=pd.read_csv(submission/"submission.csv",keep_default_na=False)
    if list(result.columns)!=["audio_filename","transcript"] or len(result)!=len(dev) or result.audio_filename.tolist()!=dev.audio_filename.tolist(): raise ValueError("runtime output failed manifest checks")
    if result.transcript.isna().any(): raise ValueError("runtime output contains NaN")
    subprocess.run(["uv","run",str(runtime/"score.py"),str(reference),"--predicted-path",str(submission/"submission.csv")],check=True)
    if sha(target)!=expected or sha(archive)!=expected: raise ValueError("tested ZIP changed")
    print("tested_sha256",expected)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--runtime",type=Path,required=True);p.add_argument("--zip",type=Path,default=ROOT/"artifacts/phase1_whisper_anchor_submission.zip");a=p.parse_args();run(a.runtime,a.zip)
