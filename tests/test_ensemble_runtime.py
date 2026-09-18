from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_src"))

from ensemble_runtime import (  # noqa: E402
    Hypothesis,
    alignment_tokens,
    edit_distance,
    official_normalize_text,
    read_submission_manifest,
    select_consensus,
    write_submission,
)


class EnsembleRuntimeTests(unittest.TestCase):
    def test_scorer_normalization_cases(self) -> None:
        self.assertEqual(official_normalize_text("Hello, WORLD!"), "hello WORLD")
        self.assertEqual(official_normalize_text("e... [laugh]"), "e...")
        self.assertEqual(official_normalize_text("foo — Bar"), "foo bar")
        self.assertEqual(alignment_tokens("(???) aku"), ("aku",))

    def test_edit_distance(self) -> None:
        self.assertEqual(edit_distance(("aku", "suka"), ("aku", "suka")), 0)
        self.assertEqual(edit_distance(("aku",), ("iya", "aku")), 1)

    def test_weighted_medoid_can_prefer_repeated_candidate(self) -> None:
        hypotheses = [
            Hypothesis("a", "aku suka makan", "mock", prior=1.0),
            Hypothesis("b", "aku suka makan", "mock", prior=1.0),
            Hypothesis("c", "aku suka tidur", "mock", prior=1.0),
        ]
        text, name, disagreement = select_consensus(hypotheses)
        self.assertEqual(text, "aku suka makan")
        self.assertIn(name, {"a", "b"})
        self.assertGreater(disagreement, 0.0)

    def test_manifest_and_csv_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clips").mkdir()
            (root / "submission_format.csv").write_text(
                "audio_filename,file_duration_seconds,language\na.mp3,5,javind\n",
                encoding="utf-8",
            )
            rows = read_submission_manifest(root)
            self.assertEqual(rows[0]["audio_filename"], "a.mp3")
            output = root / "submission" / "submission.csv"
            write_submission(
                [{"audio_filename": "a.mp3", "transcript": 'aku, "iya"'}], output
            )
            with output.open("r", encoding="utf-8", newline="") as handle:
                parsed = list(csv.DictReader(handle))
            self.assertEqual(parsed, [{"audio_filename": "a.mp3", "transcript": 'aku, "iya"'}])


if __name__ == "__main__":
    unittest.main()
