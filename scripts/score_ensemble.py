#!/usr/bin/env python3
"""Score candidate CSVs and fit heldout weighted-medoid priors.

Usage example:

    python scripts/score_ensemble.py \
      --reference processed/v2/heldout_jember.tsv \
      --predictions whisper=experiments/heldout_whisper.csv \
      --predictions qwen=experiments/heldout_qwen.csv \
      --output submission_src/calibration.json

The reference must contain ``audio_filename`` or ``clip_id`` plus a transcript
column named ``transcript`` or ``text``.  Prediction files contain
``audio_filename`` or ``clip_id`` and ``transcript``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Iterable, Mapping

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(SCRIPT_ROOT / "submission_src"))

from ensemble_runtime import Hypothesis, select_consensus  # noqa: E402


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{str(k): str(v) for k, v in row.items()} for row in csv.DictReader(handle, delimiter=delimiter)]


def _id(row: Mapping[str, str]) -> str:
    # Reference manifests can contain both columns while the prediction
    # helpers intentionally emit clip_id.  Prefer the stable clip identifier
    # whenever it is present so those tables align without renaming columns.
    for key in ("clip_id", "audio_filename"):
        if row.get(key):
            return str(row[key])
    raise ValueError("table row needs audio_filename or clip_id")


def _text(row: Mapping[str, str], reference: bool = False) -> str:
    keys = ("text", "transcript") if reference else ("transcript", "text")
    for key in keys:
        if key in row:
            return row[key]
    raise ValueError("table row needs transcript or text")


def _align(
    reference: Iterable[Mapping[str, str]], predictions: Mapping[str, list[Mapping[str, str]]]
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    ref_rows = list(reference)
    refs = {_id(row): _text(row, reference=True) for row in ref_rows}
    order = [_id(row) for row in ref_rows]
    outputs: dict[str, list[str]] = {}
    for name, rows in predictions.items():
        values = {_id(row): _text(row) for row in rows}
        missing = [key for key in order if key not in values]
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} reference rows")
        outputs[name] = [values[key] for key in order]
    return [refs[key] for key in order], outputs, order


def _wer(reference: str, predicted: str) -> tuple[int, int]:
    from ensemble_runtime import alignment_tokens, edit_distance

    ref = alignment_tokens(reference)
    pred = alignment_tokens(predicted)
    return edit_distance(ref, pred), max(len(ref), 1)


def corpus_wer(references: Iterable[str], predictions: Iterable[str]) -> float:
    errors = 0
    words = 0
    for reference, predicted in zip(references, predictions):
        error, denominator = _wer(reference, predicted)
        errors += error
        words += denominator
    return errors / max(words, 1)


def _ensemble_predictions(
    names: list[str], outputs: Mapping[str, list[str]], priors: Mapping[str, float]
) -> list[str]:
    result: list[str] = []
    for row_index in range(len(next(iter(outputs.values())))):
        hypotheses = [
            Hypothesis(
                name=name,
                text=outputs[name][row_index],
                backend="offline_prediction",
                prior=float(priors.get(name, 1.0)),
            )
            for name in names
        ]
        text, _, _ = select_consensus(hypotheses, calibration={"priors": dict(priors)})
        result.append(text)
    return result


def _grid_priors(names: list[str]) -> Iterable[dict[str, float]]:
    # A coarse grid is intentionally used: the heldout split is small and a
    # fine grid would overfit confidence noise without improving the medoid.
    levels = (0.5, 0.75, 1.0, 1.5, 2.0)
    for values in itertools.product(levels, repeat=len(names)):
        priors = dict(zip(names, values))
        max_value = max(values)
        yield {name: value / max_value for name, value in priors.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    prediction_tables: dict[str, list[dict[str, str]]] = {}
    for item in args.predictions:
        if "=" not in item:
            raise SystemExit(f"prediction must be NAME=PATH: {item}")
        name, path = item.split("=", 1)
        if not name:
            raise SystemExit("prediction name cannot be empty")
        prediction_tables[name] = _read_table(Path(path))

    references, outputs, _ = _align(_read_table(args.reference), prediction_tables)
    names = list(outputs)
    print("Individual candidate WER:")
    for name in names:
        print(f"  {name}: {corpus_wer(references, outputs[name]):.6f}")

    best_priors: dict[str, float] = {name: 1.0 for name in names}
    best_wer = corpus_wer(references, _ensemble_predictions(names, outputs, best_priors))
    for priors in _grid_priors(names):
        predictions = _ensemble_predictions(names, outputs, priors)
        score = corpus_wer(references, predictions)
        if score < best_wer:
            best_wer = score
            best_priors = priors

    print(f"Weighted-medoid WER: {best_wer:.6f}")
    print("Selected heldout priors:")
    for name in names:
        print(f"  {name}: {best_priors[name]:.3f}")

    if args.output is not None:
        payload = {
            "selection": "weighted_medoid",
            "use_quality": False,
            "priors": best_priors,
            "language_multipliers": {},
            "fit_reference": str(args.reference),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote calibration to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
