"""Run a built ZIP on a small local manifest with network access disabled."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent

def test_zip(archive: Path, count: int=2) -> dict:
    with tempfile.TemporaryDirectory(prefix="phase1-zip-smoke-") as temp:
        work=Path(temp);bundle=work/"bundle";bundle.mkdir()
        with zipfile.ZipFile(archive) as z: z.extractall(bundle)
        if not (bundle/"main.py").is_file(): raise ValueError("main.py is not at ZIP root")
        source=pd.read_csv(ROOT/"indonesian_dev/metadata.tsv",sep="\t",keep_default_na=False).head(count)
        data=work/"data";clips=data/"clips";clips.mkdir(parents=True)
        for name in source.audio_filename:
            (clips/name).symlink_to(ROOT/"indonesian_dev/clips"/name)
        source.drop(columns=["transcript"]).to_csv(data/"test_metadata.csv",index=False)
        output=work/"submission/submission.csv"
        no_network=work/"no_network";no_network.mkdir()
        (no_network/"sitecustomize.py").write_text(
            "import socket\n"
            "def blocked(*args, **kwargs):\n"
            "    raise OSError('network disabled by ZIP smoke test')\n"
            "socket.socket.connect = blocked\n"
            "socket.socket.connect_ex = blocked\n"
            "socket.create_connection = blocked\n"
            "socket.getaddrinfo = blocked\n"
        )
        env=os.environ.copy();env.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","PYTORCH_ENABLE_MPS_FALLBACK":"1","LIT_DATA_DIR":str(data),"LIT_SUBMISSION_PATH":str(output)})
        env["PYTHONPATH"]=str(no_network)+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        subprocess.run([sys.executable,str(bundle/"main.py")],cwd=work,env=env,check=True)
        result=pd.read_csv(output,keep_default_na=False)
        if list(result.columns)!=["audio_filename","transcript"] or len(result)!=len(source): raise ValueError("ZIP output schema/count mismatch")
        if result.audio_filename.tolist()!=source.audio_filename.tolist(): raise ValueError("ZIP changed manifest row order")
        if result.transcript.isna().any(): raise ValueError("ZIP output contains NaN")
        torch=__import__("torch")
        device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        return {"clips":len(result),"device":device,"network":"blocked","output_valid":True}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--zip",type=Path,default=ROOT/"artifacts/phase1_whisper_anchor_submission.zip");p.add_argument("--count",type=int,default=2);a=p.parse_args();print(test_zip(a.zip,a.count))
