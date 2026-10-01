"""Build the exact deterministic offline ZIP from a merged model."""
from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
import shutil
import os
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
CODE=[HERE/"submission/main.py", HERE/"inference/__init__.py", HERE/"inference/audio.py", HERE/"inference/whisper.py", HERE/"inference/postprocess.py"]
REQUIRED=["config.json","generation_config.json","preprocessor_config.json","tokenizer_config.json"]
TRAINING_FILES={"adapter_config.json","adapter_model.safetensors","optimizer.pt","scheduler.pt","training_args.bin","trainer_state.json"}

def build(model: Path, output: Path, language: str|None, batch_size: int=8, long_mode: str="native") -> str:
    if not model.is_dir(): raise FileNotFoundError("merged model missing")
    for name in REQUIRED:
        if not (model/name).exists(): raise FileNotFoundError(f"merged model missing {name}")
    if not any(model.glob("*.safetensors")): raise FileNotFoundError("merged weights missing")
    config={"language":language,"batch_size":batch_size,"long_mode":long_mode}
    entries={"config.json":(json.dumps(config,indent=2,sort_keys=True)+"\n").encode(),"MODEL_LICENSES.md":(HERE/"MODEL_LICENSES.md").read_bytes(),"WHISPER_LICENSE.txt":(HERE/"WHISPER_LICENSE.txt").read_bytes()}
    for path in CODE: entries[("main.py" if path.name=="main.py" else path.relative_to(HERE).as_posix())]=path.read_bytes()
    for path in sorted(model.iterdir()):
        if path.is_file() and path.name not in TRAINING_FILES|{"merge_validation.json"}:
            entries["model/"+path.name]=path
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+".tmp")
    try:
        with zipfile.ZipFile(temporary,"w") as archive:
            for name,item in sorted(entries.items()):
                info=zipfile.ZipInfo(name,(2020,1,1,0,0,0));info.external_attr=0o644<<16
                info.compress_type=zipfile.ZIP_STORED if isinstance(item,Path) and item.stat().st_size>16_000_000 else zipfile.ZIP_DEFLATED
                if isinstance(item,Path):
                    # Model weights are larger than 2 GiB. Force ZIP64 for the
                    # streamed local header, or Python raises after writing most
                    # of the checkpoint and leaves an archive without a footer.
                    with archive.open(info,"w",force_zip64=True) as dest, item.open("rb") as src: shutil.copyfileobj(src,dest,1024*1024)
                else: archive.writestr(info,item)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None: raise ValueError("ZIP CRC validation failed")
            if "main.py" not in archive.namelist() or not any(x.startswith("model/") and x.endswith(".safetensors") for x in archive.namelist()):
                raise ValueError("ZIP entrypoint or merged model missing")
        digest_hash=hashlib.sha256()
        with temporary.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b""): digest_hash.update(chunk)
        digest=digest_hash.hexdigest()
        os.replace(temporary,output)
        sidecar=output.with_suffix(output.suffix+".sha256")
        sidecar_tmp=sidecar.with_name(sidecar.name+".tmp")
        sidecar_tmp.write_text(f"{digest}  {output.name}\n")
        os.replace(sidecar_tmp,sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return digest

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",type=Path,default=HERE/"models/final_merged");p.add_argument("--output",type=Path,default=ROOT/"artifacts/phase1_whisper_anchor_submission.zip");p.add_argument("--language",choices=["indonesian","javanese","auto"],default="indonesian");p.add_argument("--batch-size",type=int,default=8);p.add_argument("--long-mode",choices=["overlap","native"],default="native");a=p.parse_args()
    print(build(a.model,a.output,None if a.language=="auto" else a.language,a.batch_size,a.long_mode))
