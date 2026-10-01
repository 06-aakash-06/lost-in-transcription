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
    DecodeSpec,
    EnsembleEngine,
    Hypothesis,
    alignment_tokens,
    canonical_manifest_language,
    edit_distance,
    official_normalize_text,
    read_submission_manifest,
    row_duration_seconds,
    rover_consensus,
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

    def test_reference_closeness_guards_anchor(self) -> None:
        hypotheses = [
            Hypothesis("qwen_id", "satu dua tiga", "mock"),
            Hypothesis("specialist", "satu dua", "mock"),
            Hypothesis("bad", "", "mock"),
        ]
        text, name, _ = select_consensus(
            hypotheses,
            calibration={
                "selection": "reference_closeness",
                "reference_closeness": {
                    "anchor": "qwen_id",
                    "candidate_names": ["specialist", "bad"],
                    "max_normalized_disagreement": 0.5,
                },
            },
        )
        self.assertEqual(name, "specialist")
        self.assertEqual(text, "satu dua")

    def test_rover_majority_fuses_words(self) -> None:
        hypotheses = [
            Hypothesis("a", "aku suka makan", "mock"),
            Hypothesis("b", "aku senang makan", "mock"),
            Hypothesis("c", "aku senang makan", "mock"),
        ]
        text = rover_consensus(
            hypotheses,
            [1.0, 1.0, 1.0],
            {"order": ["a", "b", "c"], "blank_factor": 1.0},
        )
        self.assertEqual(text, "aku senang makan")

    def test_rover_can_be_selected_by_language(self) -> None:
        hypotheses = [
            Hypothesis("a", "aku suka makan", "mock"),
            Hypothesis("b", "aku senang makan", "mock"),
            Hypothesis("c", "aku senang makan", "mock"),
        ]
        text, name, _ = select_consensus(
            hypotheses,
            calibration={
                "selection": "weighted_medoid",
                "selection_by_language": {"javind": "rover"},
                "rover": {
                    "candidate_names": ["a", "b", "c"],
                    "order": ["a", "b", "c"],
                    "blank_factor": 1.0,
                },
                "rover_by_language": {
                    "javind": {
                        "candidate_names": ["a", "b", "c"],
                        "order": ["c", "b", "a"],
                        "blank_factor": 1.0,
                    }
                },
            },
            language="javind",
        )
        self.assertEqual((text, name), ("aku senang makan", "rover"))

    def test_duration_override_can_select_full_clip_candidate(self) -> None:
        hypotheses = [
            Hypothesis("qwen_id", "satu dua tiga", "mock"),
            Hypothesis("whisper_turbo_jw", "satu", "mock"),
        ]
        calibration = {
            "selection": "weighted_medoid",
            "duration_overrides": [
                {"languages": ["javind"], "min_duration_s": 30.0, "candidate_name": "qwen_id"}
            ],
        }
        text, name, _ = select_consensus(
            hypotheses, calibration=calibration, language="javind", duration_s=31.0
        )
        self.assertEqual((text, name), ("satu dua tiga", "qwen_id"))
        self.assertEqual(row_duration_seconds({"file_duration_seconds": "31.5"}), 31.5)
        self.assertIsNone(row_duration_seconds({"file_duration_seconds": "bad"}))

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

    def test_manifest_language_aliases_and_specialist_routing(self) -> None:
        self.assertEqual(canonical_manifest_language("jv"), "jav")
        self.assertEqual(canonical_manifest_language("id-jv"), "javind")
        engine = EnsembleEngine.__new__(EnsembleEngine)
        engine.config = {}
        engine.specs = [
            DecodeSpec("qwen", "mock", "qwen", manifest_languages=("ind", "javind")),
            DecodeSpec("turbo_jw", "mock", "turbo", manifest_languages=("jav", "javind")),
            DecodeSpec("buzz_jv", "mock", "buzz", manifest_languages=("jav",)),
        ]
        self.assertEqual([item.name for item in engine._active_specs({"language": "jav"})], ["turbo_jw", "buzz_jv"])
        self.assertEqual([item.name for item in engine._active_specs({"language": "ind"})], ["qwen"])
        self.assertEqual(
            [item.name for item in engine._active_specs({"language": "javind"})],
            ["qwen", "turbo_jw"],
        )
        engine.config = {"route_language": "javind"}
        self.assertEqual(
            [item.name for item in engine._active_specs({"language": "ind"})],
            ["qwen", "turbo_jw"],
        )
        self.assertEqual(
            [item.name for item in engine._active_specs({})],
            ["qwen", "turbo_jw"],
        )

    def test_adaptive_candidate_is_decoded_in_batches(self) -> None:
        class FakeBackend:
            def __init__(self, label: str) -> None:
                self.label = label
                self.batch_sizes: list[int] = []

            def decode_many(self, audio_sources, specs):
                self.batch_sizes.append(len(audio_sources))
                return [Hypothesis(spec.name, self.label, "fake") for spec in specs]

        base = [
            DecodeSpec("qwen", "fake", "qwen", manifest_languages=("javind",)),
            DecodeSpec("turbo", "fake", "turbo", manifest_languages=("javind",)),
        ]
        extra = DecodeSpec("buzz", "fake", "buzz", manifest_languages=("javind",))
        engine = EnsembleEngine.__new__(EnsembleEngine)
        engine.config = {
            "inference_batch_size": 2,
            "adaptive_trigger": {
                "min_disagreement": 0.0,
                "min_duration_s": 999.0,
                "max_extra_decodes": 1,
            },
        }
        engine.specs = base
        engine.extra_specs = [extra]
        engine.audio_preprocessing = {}
        engine.calibration = {}
        qwen = FakeBackend("first")
        turbo = FakeBackend("second")
        buzz = FakeBackend("specialist")
        engine._backends = {
            ("fake", "qwen", False): qwen,
            ("fake", "turbo", False): turbo,
            ("fake", "buzz", False): buzz,
        }

        outputs = engine.transcribe_batch(
            [Path("a.wav"), Path("b.wav"), Path("c.wav")],
            [{"language": "javind"}] * 3,
        )
        self.assertEqual(len(outputs), 3)
        self.assertEqual(buzz.batch_sizes, [2, 1])


if __name__ == "__main__":
    unittest.main()
