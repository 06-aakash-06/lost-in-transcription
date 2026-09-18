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
    if configured.parent.exists() or str(configured).startswith("/code_execution"):
        return configured
    return Path.cwd() / "submission" / "submission.csv"


def main() -> None:
    bundle_root = Path(__file__).resolve().parent
    data_dir = _data_dir()
    clips_dir = data_dir / "clips"
    rows = read_submission_manifest(data_dir)
    engine = EnsembleEngine(bundle_root)

    predictions: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        audio_path = clips_dir / row["audio_filename"]
        if not audio_path.is_file():
            raise FileNotFoundError(f"manifest audio file is missing at row {index}")
        try:
            transcript = engine.transcribe(audio_path, row)
        except Exception as exc:
            # Do not include sample names or transcript content in logs.
            raise RuntimeError(f"inference failed at manifest row {index}") from exc
        predictions.append(
            {
                "audio_filename": row["audio_filename"],
                "transcript": transcript,
            }
        )
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"Processed {index}/{len(rows)} clips")

    output_path = _output_path()
    write_submission(predictions, output_path)
    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
