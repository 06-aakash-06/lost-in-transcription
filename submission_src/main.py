"""Competition entry point.

The archive builder places this file at the ZIP root.  The evaluator supplies
the read-only ``data`` directory and expects exactly
``submission/submission.csv`` with ``audio_filename`` and ``transcript``.
"""

from __future__ import annotations

import os
from pathlib import Path

from ensemble_runtime import EnsembleEngine, read_submission_manifest, write_submission


def _data_dir() -> Path:
    configured = Path(os.environ.get("LIT_DATA_DIR", "/code_execution/data"))
    if configured.exists():
        return configured
    return Path.cwd() / "data"


def _output_path() -> Path:
    configured = Path(os.environ.get("LIT_SUBMISSION_PATH", "/code_execution/submission/submission.csv"))
    if "LIT_SUBMISSION_PATH" in os.environ:
        return configured
    if configured.parent.exists() or str(configured).startswith("/code_execution"):
        return configured
    return Path.cwd() / "submission" / "submission.csv"


def main() -> None:
    bundle_root = Path(__file__).resolve().parent
    data_dir = _data_dir()
    clips_dir = data_dir / "clips"
    rows = read_submission_manifest(data_dir)
    engine = EnsembleEngine(bundle_root)
    batch_size = max(1, int(engine.config.get("inference_batch_size", 8)))

    predictions: list[dict[str, str]] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_paths = [clips_dir / row["audio_filename"] for row in batch_rows]
        for offset, audio_path in enumerate(batch_paths):
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"manifest audio file is missing at row {start + offset + 1}"
                )
        try:
            transcripts = engine.transcribe_batch(batch_paths, batch_rows)
        except Exception as exc:
            # Do not include sample names or transcript content in logs.
            raise RuntimeError(f"inference failed near manifest row {start + 1}") from exc
        predictions.extend(
            {
                "audio_filename": row["audio_filename"],
                "transcript": transcript,
            }
            for row, transcript in zip(batch_rows, transcripts)
        )
        processed = min(start + len(batch_rows), len(rows))
        if processed == len(rows) or processed % 25 == 0 or start == 0:
            print(f"Processed {processed}/{len(rows)} clips")

    output_path = _output_path()
    write_submission(predictions, output_path)
    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
