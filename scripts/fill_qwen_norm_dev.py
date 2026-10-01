#!/usr/bin/env python3
"""Complete the existing normalized-Qwen dev cache using bundled weights."""

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_src"))
from audio_preprocess import prepare_audio  # noqa: E402
from ensemble_runtime import DecodeSpec, QwenASRBackend  # noqa: E402


def main():
    with (ROOT / "processed/v2/dev.tsv").open(newline="", encoding="utf-8") as handle:
        dev = list(csv.DictReader(handle, delimiter="\t"))
    cache = ROOT / "experiments/dev_qwen_norm_full.csv"
    with cache.open(newline="", encoding="utf-8") as handle:
        seen = {row["clip_id"] for row in csv.DictReader(handle)}
    missing = [row for row in dev if row["clip_id"] not in seen]
    missing = missing[int(os.environ.get("LIT_SKIP_FIRST", "0")):]
    limit = int(os.environ.get("LIT_MAX_CLIPS", "0"))
    if limit > 0:
        missing = missing[:limit]
    if not missing:
        return
    config = json.loads((ROOT / "submission_src/ensemble_config.json").read_text())
    spec = DecodeSpec.from_mapping(next(x for x in config["decodes"] if x["name"] == "qwen_id"))
    profile = config["audio_preprocessing"]["profiles"][spec.preprocess_profile]
    batch_size = int(os.environ.get("LIT_BATCH_SIZE", "2"))
    backend = QwenASRBackend(ROOT / "submission_src" / spec.model, max_new_tokens=spec.max_new_tokens, device=os.environ.get("LIT_DEVICE", "mps"), max_inference_batch_size=batch_size)
    with cache.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "transcript"])
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start+batch_size]
            audio = [prepare_audio(ROOT / "processed/v2" / row["path"], profile) for row in batch]
            results = backend.decode_many(audio, [spec]*len(batch))
            for row, result in zip(batch, results):
                writer.writerow({"clip_id": row["clip_id"], "transcript": result.cleaned_text})
            handle.flush()
            print(f"qwen norm: {min(start+batch_size,len(missing))}/{len(missing)}", flush=True)


if __name__ == "__main__":
    main()
