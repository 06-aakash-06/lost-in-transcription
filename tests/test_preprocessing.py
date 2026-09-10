from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lostintranscription_preproc import (  # noqa: E402
    DEFAULT_CONFIG,
    _quality_weight,
    loudness_normalize,
    merge_segments,
    normalize_training_text,
    official_normalize_text,
    parse_timestamp,
    vad_edges,
)


class PreprocessingUnitTests(unittest.TestCase):
    def test_timestamp_forms(self) -> None:
        self.assertAlmostEqual(parse_timestamp("01:02:03.5"), 3723.5)
        self.assertAlmostEqual(parse_timestamp("02:03.5"), 123.5)
        self.assertAlmostEqual(parse_timestamp(4), 4.0)
        with self.assertRaises(ValueError):
            parse_timestamp("-1")

    def test_training_normalizer_preserves_scored_structure(self) -> None:
        value = normalize_training_text("Těrus — “nèk”…  goed\tword")
        self.assertEqual(value, 'Terus "nek"... goed word')
        self.assertIn("...", value)
        self.assertNotIn("—", value)

    def test_official_normalizer_regressions(self) -> None:
        cases = {
            "Hello, WORLD!": "hello WORLD",
            "e... [laugh]": "e...",
            "foo — Bar": "foo bar",
            "(???) aku": "aku",
            "A.B": "a b",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(official_normalize_text(raw), expected)

    def test_loudness_respects_peak_ceiling(self) -> None:
        cfg = dict(DEFAULT_CONFIG)
        audio = np.full(16000, 0.01, dtype=np.float32)
        output, gain_db = loudness_normalize(audio, cfg)
        ceiling = 10 ** (cfg["peak_ceiling_dbfs"] / 20)
        self.assertGreater(gain_db, 0)
        self.assertLessEqual(float(np.max(np.abs(output))), ceiling + 1e-6)

    def test_vad_is_edge_only(self) -> None:
        cfg = dict(DEFAULT_CONFIG)
        audio = np.zeros(16000, dtype=np.float32)
        audio[8000:9600] = 0.1
        start, end = vad_edges(audio, -80.0, cfg)
        self.assertGreater(start, 0)
        self.assertLess(end, len(audio))
        self.assertGreaterEqual(start, len(audio) - end - 1)

    def test_merge_partitions_source_rows(self) -> None:
        rows = pd.DataFrame(
            [
                {"source_row": 10, "session": 1, "start_s": 0.0, "end_s": 5.0, "text_norm": "a"},
                {"source_row": 11, "session": 1, "start_s": 5.0, "end_s": 10.0, "text_norm": "b"},
                {"source_row": 12, "session": 1, "start_s": 10.0, "end_s": 15.0, "text_norm": "c"},
            ]
        )
        chunks = merge_segments(rows, {10: -20.0, 11: -20.0, 12: -20.0}, DEFAULT_CONFIG)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks.iloc[0]["source_rows"], [10, 11, 12])
        self.assertEqual(chunks.iloc[0]["text"], "a b c")

    def test_quality_weight_is_bounded(self) -> None:
        tier, weight = _quality_weight(
            {
                "flag_offmic": True,
                "flag_clipped_batch": True,
                "flag_low_snr": True,
                "pre_silence_frac": 0.9,
            }
        )
        self.assertEqual(tier, "BRONZE")
        self.assertGreaterEqual(weight, 0.5)
        self.assertLessEqual(weight, 1.0)


if __name__ == "__main__":
    unittest.main()
