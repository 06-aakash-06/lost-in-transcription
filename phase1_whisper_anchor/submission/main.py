"""Official offline submission entrypoint."""
from __future__ import annotations
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import json
import time
from pathlib import Path
import pandas as pd
from inference.audio import load_audio
from inference.whisper import WhisperAnchor

def validate_output(metadata: pd.DataFrame, output: pd.DataFrame) -> None:
    if list(output.columns) != ["audio_filename", "transcript"]: raise ValueError("incorrect output columns")
    if len(output) != len(metadata): raise ValueError("incorrect row count")
    if output.audio_filename.tolist() != metadata.audio_filename.tolist(): raise ValueError("filename order changed")
    if output.transcript.isna().any() or not output.transcript.map(lambda x: isinstance(x,str)).all(): raise ValueError("invalid transcripts")

def run(data_dir: Path, output_path: Path, model_dir: Path, config_path: Path) -> None:
    start = time.monotonic()
    print("starting inference", flush=True)
    config = json.loads(config_path.read_text())
    metadata = pd.read_csv(data_dir / "test_metadata.csv", dtype={"audio_filename":str}, keep_default_na=False)
    if "audio_filename" not in metadata or metadata.audio_filename.isna().any(): raise ValueError("invalid metadata")
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("device=" + device, flush=True)
    anchor = WhisperAnchor(model_dir, config.get("language"), int(config.get("batch_size",8)), config.get("long_mode","native"))
    print("model loaded", flush=True)
    clips = data_dir / "clips"
    texts = []
    batch_size = int(config.get("batch_size",8))
    for start_idx in range(0,len(metadata),batch_size):
        names = metadata.audio_filename.iloc[start_idx:start_idx+batch_size]
        waves = []
        for name in names:
            # Reject path traversal even though the manifest normally uses basenames.
            if Path(name).name != name: raise ValueError("invalid manifest audio path")
            waves.append(load_audio(clips / name, redact_errors=True))
        texts.extend(anchor.transcribe_arrays(waves))
    output = pd.DataFrame({"audio_filename": metadata.audio_filename.tolist(), "transcript": texts})
    validate_output(metadata, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print("inference complete", flush=True)
    print("submission written", flush=True)
    print(f"elapsed={time.monotonic()-start:.1f}", flush=True)

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run(Path(os.environ.get("LIT_DATA_DIR","/code_execution/data")), Path(os.environ.get("LIT_SUBMISSION_PATH","/code_execution/submission/submission.csv")), here / "model", here / "config.json")
