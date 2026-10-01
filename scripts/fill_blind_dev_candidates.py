#!/usr/bin/env python3
"""Decode only dev rows missing from cached mixed-route candidate CSVs."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_src"))
from ensemble_runtime import DecodeSpec, TransformersWhisperBackend  # noqa: E402


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    with (ROOT / "processed/v2/dev.tsv").open(newline="", encoding="utf-8") as handle:
        dev = list(csv.DictReader(handle, delimiter="\t"))
    config = json.loads((ROOT / "submission_src/ensemble_config.json").read_text())
    definitions = config["decodes"] + config["adaptive_decodes"]
    for name, cached_file in (
        ("whisper_turbo_jw", "dev_whisper_turbo_jw_current_full.csv"),
        ("buzz_javanese", "dev_buzz_jv_full.csv"),
    ):
        spec = DecodeSpec.from_mapping(next(x for x in definitions if x["name"] == name))
        seen = {x["clip_id"] for x in read_csv(ROOT / "experiments" / cached_file)}
        missing = [row for row in dev if row["clip_id"] not in seen]
        path = ROOT / "experiments" / f"dev_{name}_blind_missing.csv"
        if path.exists():
            seen = {x["clip_id"] for x in read_csv(path)}
            missing = [row for row in missing if row["clip_id"] not in seen]
        if not missing:
            continue
        print(f"{name}: decoding {len(missing)} missing clips", flush=True)
        backend = TransformersWhisperBackend(ROOT / "submission_src" / spec.model, device="mps")
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["clip_id", "transcript"])
            if handle.tell() == 0:
                writer.writeheader()
            for index, row in enumerate(missing, 1):
                audio = ROOT / "processed/v2" / row["path"]
                prediction = backend.decode(audio, spec)
                writer.writerow({"clip_id": row["clip_id"], "transcript": prediction.cleaned_text})
                handle.flush()
                if index % 5 == 0 or index == len(missing):
                    print(f"{name}: {index}/{len(missing)}", flush=True)
        del backend


if __name__ == "__main__":
    main()
