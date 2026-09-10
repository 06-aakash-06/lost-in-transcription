#!/usr/bin/env python3
"""Validate a built preprocessing artifact before it is used for training."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lostintranscription_preproc import official_normalize_text  # noqa: E402


def _text_or_empty(value: object) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def _check_audio(frame: pd.DataFrame, out: Path, label: str, errors: list[str]) -> None:
    for row in frame.itertuples(index=False):
        relative = Path(str(row.path))
        path = out / relative
        try:
            info = sf.info(str(path))
        except Exception as exc:  # pragma: no cover - exercised on corrupt artifacts
            errors.append(f"{label} {relative} is unreadable: {exc}")
            continue
        if info.samplerate != 16000 or info.channels != 1 or info.format != "FLAC" or info.subtype != "PCM_16":
            errors.append(
                f"{label} {relative} has {info.samplerate} Hz/{info.channels} ch/"
                f"{info.format}/{info.subtype}, expected 16000 Hz/mono/FLAC/PCM_16"
            )
        actual_duration = info.frames / info.samplerate
        if not np.isclose(actual_duration, float(row.duration_s), atol=1.0 / info.samplerate + 1e-5):
            errors.append(
                f"{label} {relative} duration metadata {row.duration_s} != actual {actual_duration:.6f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--version", default="v2")
    args = parser.parse_args()

    root = args.root.resolve()
    out = root / "processed" / args.version
    errors: list[str] = []
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        print(f"FAIL: missing {manifest_path}")
        return 1

    manifest = json.loads(manifest_path.read_text())
    train_path = out / "train_jember.tsv"
    train_only_path = out / "train_jember_trainonly.tsv"
    heldout_path = out / "heldout_jember.tsv"
    dev_path = out / "dev.tsv"
    prefilter_path = out / "meta" / "train_chunks_all_prefilter.tsv"
    integrity_path = out / "meta" / "jember_rows_postintegrity.tsv"
    required_files = [train_path, train_only_path, heldout_path, dev_path, prefilter_path, integrity_path]
    for path in required_files:
        if not path.exists():
            errors.append(f"missing required artifact: {path.relative_to(out)}")
    if errors:
        print("VALIDATION FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1

    train = pd.read_csv(train_path, sep="\t")
    train_only = pd.read_csv(train_only_path, sep="\t")
    heldout = pd.read_csv(heldout_path, sep="\t")
    dev = pd.read_csv(dev_path, sep="\t")
    prefilter = pd.read_csv(prefilter_path, sep="\t")
    integrity = pd.read_csv(integrity_path, sep="\t")

    for label, frame, columns in [
        ("train", train, ["clip_id", "path", "session", "split", "text", "duration_s", "source_rows"]),
        ("dev", dev, ["clip_id", "path", "audio_filename", "text", "duration_s"]),
        ("prefilter", prefilter, ["clip_id", "path", "session", "drop_reason", "split", "source_rows"]),
        ("integrity", integrity, ["source_row", "session", "text_norm"]),
    ]:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            errors.append(f"{label} missing columns: {missing}")

    if not errors:
        expected_train = int(manifest["outputs"]["train"]["chunks"])
        expected_dev = int(manifest["outputs"]["dev"]["clips"])
        if len(train) != expected_train:
            errors.append(f"train count {len(train)} != manifest {expected_train}")
        if len(dev) != expected_dev:
            errors.append(f"dev count {len(dev)} != manifest {expected_dev}")
        if len(train_only) + len(heldout) != len(train):
            errors.append("train-only plus heldout rows does not equal train rows")
        if set(train_only["split"].astype(str)) != {"train"}:
            errors.append("train_jember_trainonly.tsv contains a non-train split")
        if set(heldout["split"].astype(str)) != {"heldout"}:
            errors.append("heldout_jember.tsv contains a non-heldout split")
        if train["clip_id"].duplicated().any() or dev["clip_id"].duplicated().any():
            errors.append("duplicate clip_id in a shipped manifest")
        if train["path"].duplicated().any() or dev["path"].duplicated().any():
            errors.append("duplicate audio path in a shipped manifest")

        train_sessions = set(train_only["session"].astype(int))
        heldout_sessions = set(heldout["session"].astype(int))
        if train_sessions & heldout_sessions:
            errors.append("session leakage between train and heldout splits")
        manifest_heldout = set(int(value) for value in manifest.get("heldout_sessions", []))
        if heldout_sessions != manifest_heldout:
            errors.append("heldout session list does not match manifest")
        if set(train["split"].astype(str)) - {"train", "heldout"}:
            errors.append("unexpected split label in train manifest")
        if train["text"].map(_text_or_empty).str.strip().eq("").any():
            errors.append("empty transcript in shipped training manifest")
        if prefilter["drop_reason"].isna().any():
            errors.append("prefilter manifest has an implicit/blank drop reason; expected KEEP or an explicit reason")
        dropped = prefilter[prefilter["drop_reason"].astype(str) != "KEEP"]
        if not dropped.empty and set(dropped["split"].astype(str)) != {"excluded"}:
            errors.append("quality-gated rows are not marked excluded")
        if train["duration_s"].lt(2.0).any() or train["duration_s"].gt(30.0).any():
            errors.append("shipped training duration is outside [2, 30] seconds")
        if dev["duration_s"].lt(2.0).any() or dev["duration_s"].gt(40.0).any():
            errors.append("processed dev duration is outside the competition [2, 40] second range")

        for label, frame in [("train", train), ("dev", dev)]:
            paths = [Path(str(value)) for value in frame["path"]]
            if any(path.is_absolute() or ".." in path.parts for path in paths):
                errors.append(f"{label} contains an unsafe relative audio path")
            if any(not (out / path).exists() for path in paths):
                errors.append(f"{label} has a missing referenced audio file")

        _check_audio(train, out, "train", errors)
        _check_audio(dev, out, "dev", errors)

        source_rows: list[int] = []
        for value in prefilter["source_rows"]:
            try:
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    raise ValueError("not a list")
                source_rows.extend(int(item) for item in parsed)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"invalid source_rows JSON: {exc}")
        expected_rows = set(integrity["source_row"].astype(int))
        if len(source_rows) != len(set(source_rows)):
            errors.append("source rows are duplicated or merged more than once")
        if set(source_rows) != expected_rows:
            errors.append("prefilter chunks do not partition the post-integrity source rows")

        combining = [
            (clip_id, char)
            for clip_id, text in zip(train["clip_id"], train["text"])
            for char in str(text)
            if unicodedata.category(char) == "Mn"
        ]
        if combining:
            errors.append(f"training text still contains combining marks, first={combining[0]}")
        if train["text"].astype(str).str.contains("—", regex=False).any():
            errors.append("training text still contains an em dash")

        source_dev = pd.read_csv(root / "indonesian_dev" / "metadata.tsv", sep="\t")
        expected_refs = {
            str(row.audio_filename): _text_or_empty(row.transcript)
            for row in source_dev.itertuples(index=False)
        }
        actual_refs = dict(zip(dev["audio_filename"].astype(str), dev["text"].map(_text_or_empty)))
        if actual_refs != expected_refs:
            errors.append("dev references are not verbatim copies of the source metadata")

        # A few scorer-regression cases protect the local metric from an
        # accidental broad lowercase/punctuation rewrite.
        scorer_cases = {
            "Hello, WORLD!": "hello WORLD",
            "e... [laugh]": "e...",
            "foo — Bar": "foo bar",
            "(???) aku": "aku",
        }
        for raw, expected in scorer_cases.items():
            if official_normalize_text(raw) != expected:
                errors.append(f"official normalizer regression for {raw!r}")

    if errors:
        print("VALIDATION FAILED")
        print("\n".join(f"- {error}" for error in errors[:40]))
        if len(errors) > 40:
            print(f"- ... and {len(errors) - 40} more")
        return 1

    print(f"VALIDATION PASSED: {out}")
    print(f"  train={len(train)} ({len(train_only)} train + {len(heldout)} heldout), dev={len(dev)}")
    print(f"  audio={len(train) + len(dev)} files; sessions held out={sorted(heldout_sessions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
