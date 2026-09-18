#!/usr/bin/env python3
"""Run one bundled/local ASR candidate on a reference table.

This is an evaluation helper only.  It reads the session-disjoint heldout or
the validation table and writes ``clip_id,transcript`` predictions that can be
passed to ``scripts/score_ensemble.py``.  It never reads the competition test
set and it has no network-dependent inference path.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_src"))

from ensemble_runtime import (  # noqa: E402
    DecodeSpec,
    FasterWhisperBackend,
    QwenASRBackend,
    TransformersCTCBackend,
)


def _read_rows(path: Path, max_rows: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        raise ValueError(f"reference table is empty: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--backend", choices=("ctc", "whisper", "qwen"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language", default="id")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    rows = _read_rows(args.reference, args.max_rows)
    if args.backend == "ctc":
        backend = TransformersCTCBackend(
            args.model,
            device=args.device,
            use_language_model=False,
        )
        spec = DecodeSpec(
            name=args.name,
            backend="transformers_ctc",
            model=str(args.model),
        )
    elif args.backend == "whisper":
        backend = FasterWhisperBackend(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=8,
            num_workers=1,
        )
        spec = DecodeSpec(
            name=args.name,
            backend="faster_whisper",
            model=str(args.model),
            language=args.language or None,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
        )
    else:
        backend = QwenASRBackend(
            args.model,
            max_new_tokens=256,
            device=args.device,
        )
        spec = DecodeSpec(
            name=args.name,
            backend="qwen_asr",
            model=str(args.model),
            language=args.language or None,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "transcript"], lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            audio = args.audio_root / row["path"]
            if not audio.is_file():
                raise FileNotFoundError(audio)
            result = backend.decode(audio, spec)
            writer.writerow({"clip_id": row["clip_id"], "transcript": result.cleaned_text})
            if index == 1 or index % 10 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)}", flush=True)
    print(f"Wrote {len(rows)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
