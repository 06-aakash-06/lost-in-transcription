"""Offline ASR ensemble runtime used by the competition submission.

The module is intentionally dependency-light at import time.  The optional ASR
packages are imported only when their configured backend is instantiated.  This
keeps the local evaluation and consensus code testable on a CPU-only machine
while the packaged submission uses the competition runtime's GPU dependencies.

The ensemble is transcript-level rather than logit-level because the candidate
models use different tokenizers.  A weighted medoid is a deliberately
conservative minimum-risk approximation: it can select the candidate closest to
the other candidates without inventing a new transcript through an unvalidated
word voter.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# This mirrors the official scorer's normalization.  It is used only to align
# candidates for consensus and diagnostics.  The raw selected transcript is
# written to the submission so the scorer remains the only authority on final
# normalization.
_BRACKETED_RE = re.compile(r"\[[^\]]+\]")
_UNINTELLIGIBLE_PAREN_RE = re.compile(r"\(\?+\)")
_WORD_PAREN_RE = re.compile(r"\(([^()]*)\)")
_PUNCTUATION_OTHER_RE = re.compile('[¿¡\";:]+')
_COMMA_RE = re.compile(",+")
_SENTENCE_INITIAL_RE = re.compile(r"(^\s*|[.!?—]\s*)([^\W\d_])([^\W\d_]?)")
_SENTENCE_END_RE = re.compile("[!?]+")
_MULTISPACE_RE = re.compile("  +")


def _lowercase_sentence_initial(match: re.Match[str]) -> str:
    delimiter, first, second = match.group(1), match.group(2), match.group(3)
    if first.isupper() and not (second and second.isupper()):
        first = first.lower()
    return delimiter + first + second


def official_normalize_text(value: Any) -> str:
    """Return the competition-normalized text used for candidate alignment."""

    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    text = text.replace("~", "")
    text = _BRACKETED_RE.sub(" ", text)
    text = _UNINTELLIGIBLE_PAREN_RE.sub(" ", text)
    text = _WORD_PAREN_RE.sub(r"\1", text)
    text = text.replace("#x27;", "'")
    text = _PUNCTUATION_OTHER_RE.sub(" ", text)
    text = _SENTENCE_INITIAL_RE.sub(_lowercase_sentence_initial, text)
    text = text.replace("—", ", ")
    text = _COMMA_RE.sub(" ", text)
    text = _SENTENCE_END_RE.sub(" ", text)
    text = text.replace("...", "!ELLIPSIS!").replace(".", " ").replace(
        "!ELLIPSIS!", "..."
    )
    while " ... " in text:
        text = text.replace(" ... ", " ")
    return _MULTISPACE_RE.sub(" ", text).strip()


def alignment_tokens(value: Any) -> tuple[str, ...]:
    return tuple(official_normalize_text(value).split())


def clean_transcript(value: Any) -> str:
    """Remove transport-only whitespace/control noise, preserving free text."""

    if value is None:
        return ""
    text = str(value).replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    return " ".join(text.split()).strip()


def edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """Levenshtein distance with O(min(len(left), len(right))) memory."""

    if len(left) < len(right):
        short, long = left, right
    else:
        short, long = right, left
    previous = list(range(len(short) + 1))
    for i, long_token in enumerate(long, start=1):
        current = [i]
        for j, short_token in enumerate(short, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (long_token != short_token),
                )
            )
        previous = current
    return previous[-1]


def normalized_distance(left: Sequence[str], right: Sequence[str]) -> float:
    denominator = max(len(left), len(right), 1)
    return edit_distance(left, right) / denominator


@dataclass(frozen=True)
class DecodeSpec:
    """One deterministic model/decode candidate."""

    name: str
    backend: str
    model: str
    language: str | None = None
    beam_size: int = 5
    temperature: float = 0.0
    vad_filter: bool = False
    condition_on_previous_text: bool = False
    max_new_tokens: int = 256
    use_language_model: bool = False
    lm_alpha: float = 0.5
    lm_beta: float = 1.5
    prior: float = 1.0
    enabled: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DecodeSpec":
        return cls(
            name=str(value["name"]),
            backend=str(value["backend"]),
            model=str(value["model"]),
            language=None if value.get("language") in (None, "", "auto") else str(value["language"]),
            beam_size=int(value.get("beam_size", 5)),
            temperature=float(value.get("temperature", 0.0)),
            vad_filter=bool(value.get("vad_filter", False)),
            condition_on_previous_text=bool(value.get("condition_on_previous_text", False)),
            max_new_tokens=int(value.get("max_new_tokens", 256)),
            use_language_model=bool(value.get("use_language_model", False)),
            lm_alpha=float(value.get("lm_alpha", 0.5)),
            lm_beta=float(value.get("lm_beta", 1.5)),
            prior=float(value.get("prior", 1.0)),
            enabled=bool(value.get("enabled", True)),
        )


@dataclass
class Hypothesis:
    """A candidate transcript and model-side metadata."""

    name: str
    text: str
    backend: str
    prior: float = 1.0
    quality: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cleaned_text(self) -> str:
        return clean_transcript(self.text)

    @property
    def tokens(self) -> tuple[str, ...]:
        return alignment_tokens(self.cleaned_text)


def _quality_multiplier(hypothesis: Hypothesis, use_quality: bool) -> float:
    """Use ASR diagnostics only when calibration explicitly enables them."""

    if not use_quality:
        return 1.0
    value = float(hypothesis.quality)
    if not math.isfinite(value):
        return 1.0
    return max(0.05, min(2.0, value))


def _language_multiplier(
    calibration: Mapping[str, Any], candidate_name: str, language: str | None
) -> float:
    if not language:
        return 1.0
    by_language = calibration.get("language_multipliers", {})
    if not isinstance(by_language, Mapping):
        return 1.0
    values = by_language.get(str(language), {})
    if not isinstance(values, Mapping):
        return 1.0
    try:
        value = float(values.get(candidate_name, 1.0))
    except (TypeError, ValueError):
        return 1.0
    return max(0.05, min(20.0, value))


def candidate_weights(
    hypotheses: Sequence[Hypothesis],
    calibration: Mapping[str, Any] | None = None,
    language: str | None = None,
) -> list[float]:
    """Return positive weights for reference-free candidate consensus.

    ``calibration`` is produced from predictions on the session-disjoint
    heldout split.  An empty calibration is safe and gives every candidate its
    configured prior.  Empty transcripts receive a small penalty so a model
    that abstains cannot win simply because it is close to another empty
    hypothesis.
    """

    calibration = calibration or {}
    configured_priors = calibration.get("priors", {})
    if not isinstance(configured_priors, Mapping):
        configured_priors = {}
    use_quality = bool(calibration.get("use_quality", False))
    weights: list[float] = []
    for hypothesis in hypotheses:
        try:
            prior = float(configured_priors.get(hypothesis.name, hypothesis.prior))
        except (TypeError, ValueError):
            prior = hypothesis.prior
        weight = max(0.01, prior)
        weight *= _language_multiplier(calibration, hypothesis.name, language)
        weight *= _quality_multiplier(hypothesis, use_quality)
        if not hypothesis.tokens:
            weight *= 0.25
        weights.append(max(0.001, weight))
    return weights


def weighted_medoid_index(
    hypotheses: Sequence[Hypothesis], weights: Sequence[float]
) -> int:
    """Choose the candidate minimizing weighted expected normalized edit cost."""

    if not hypotheses:
        raise ValueError("Cannot select a medoid from zero hypotheses")
    if len(hypotheses) != len(weights):
        raise ValueError("hypotheses and weights must have the same length")
    token_lists = [hypothesis.tokens for hypothesis in hypotheses]
    best_index = 0
    best_key: tuple[float, float, int] | None = None
    for i, tokens in enumerate(token_lists):
        expected_cost = 0.0
        for j, other in enumerate(token_lists):
            expected_cost += float(weights[j]) * normalized_distance(tokens, other)
        # Stable ties prefer the candidate with the stronger calibrated weight,
        # then the earlier configured candidate.
        key = (expected_cost, -float(weights[i]), i)
        if best_key is None or key < best_key:
            best_key = key
            best_index = i
    return best_index


def mean_pairwise_disagreement(hypotheses: Sequence[Hypothesis]) -> float:
    if len(hypotheses) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(hypotheses)):
        for j in range(i + 1, len(hypotheses)):
            total += normalized_distance(hypotheses[i].tokens, hypotheses[j].tokens)
            count += 1
    return total / max(count, 1)


def select_consensus(
    hypotheses: Sequence[Hypothesis],
    calibration: Mapping[str, Any] | None = None,
    language: str | None = None,
) -> tuple[str, str, float]:
    """Return ``(raw_text, selected_name, disagreement)``."""

    if not hypotheses:
        raise ValueError("No ASR hypotheses were generated")
    weights = candidate_weights(hypotheses, calibration=calibration, language=language)
    index = weighted_medoid_index(hypotheses, weights)
    disagreement = mean_pairwise_disagreement(hypotheses)
    selected = hypotheses[index]
    return selected.cleaned_text, selected.name, disagreement


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def whisper_quality(info: Any, segments: Sequence[Any]) -> float:
    """Convert faster-whisper diagnostics into a bounded optional quality hint."""

    avg_logprob_values = [_finite_or_none(getattr(item, "avg_logprob", None)) for item in segments]
    avg_logprob_values = [value for value in avg_logprob_values if value is not None]
    compression_values = [_finite_or_none(getattr(item, "compression_ratio", None)) for item in segments]
    compression_values = [value for value in compression_values if value is not None]
    no_speech_values = [_finite_or_none(getattr(item, "no_speech_prob", None)) for item in segments]
    no_speech_values = [value for value in no_speech_values if value is not None]

    quality = 1.0
    if avg_logprob_values:
        # -0.2 is a strong segment score; -1.5 is weak.  This is only a
        # bounded feature, not a replacement for heldout calibration.
        quality *= max(0.35, min(1.25, (sum(avg_logprob_values) / len(avg_logprob_values) + 1.5) / 1.3))
    if compression_values:
        worst = max(compression_values)
        if worst > 2.4:
            quality *= max(0.25, 2.4 / worst)
    if no_speech_values and not any(alignment_tokens(getattr(item, "text", "")) for item in segments):
        quality *= max(0.20, 1.0 - max(no_speech_values))
    return max(0.05, min(2.0, quality))


class FasterWhisperBackend:
    """Lazy wrapper around ``faster-whisper``."""

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        compute_type: str = "float16",
        cpu_threads: int = 8,
        num_workers: int = 1,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - runtime-only branch
            raise RuntimeError("faster-whisper is required for a Whisper candidate") from exc
        self.model = WhisperModel(
            str(model_path),
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

    def decode(self, audio_path: Path, spec: DecodeSpec) -> Hypothesis:
        segments, info = self.model.transcribe(
            str(audio_path),
            language=spec.language,
            task="transcribe",
            beam_size=spec.beam_size,
            temperature=spec.temperature,
            condition_on_previous_text=spec.condition_on_previous_text,
            vad_filter=spec.vad_filter,
            word_timestamps=False,
            max_new_tokens=spec.max_new_tokens,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        segment_list = list(segments)
        text = clean_transcript(" ".join(clean_transcript(getattr(item, "text", "")) for item in segment_list))
        metadata = {
            "language": getattr(info, "language", None),
            "language_probability": _finite_or_none(getattr(info, "language_probability", None)),
            "n_segments": len(segment_list),
        }
        return Hypothesis(
            name=spec.name,
            text=text,
            backend=spec.backend,
            prior=spec.prior,
            quality=whisper_quality(info, segment_list),
            metadata=metadata,
        )


def _qwen_language(language: str | None) -> str | None:
    if language is None:
        return None
    aliases = {
        "id": "Indonesian",
        "ind": "Indonesian",
        "indonesian": "Indonesian",
    }
    return aliases.get(language.lower(), language)


class QwenASRBackend:
    """Lazy wrapper around the official local Qwen3-ASR transformers backend."""

    def __init__(
        self,
        model_path: Path,
        max_new_tokens: int = 256,
        device: str = "cuda",
    ) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:  # pragma: no cover - runtime-only branch
            raise RuntimeError("torch and qwen-asr are required for a Qwen candidate") from exc
        requested = str(device).lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            device_map = requested if ":" in requested else "cuda:0"
            dtype = torch.bfloat16
        elif requested == "mps" and torch.backends.mps.is_available():
            device_map = "mps"
            dtype = torch.float16
        else:
            device_map = "cpu"
            dtype = torch.float32
        self.model = Qwen3ASRModel.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=1,
            max_new_tokens=max_new_tokens,
        )

    def decode(self, audio_path: Path, spec: DecodeSpec) -> Hypothesis:
        results = self.model.transcribe(
            audio=str(audio_path),
            language=_qwen_language(spec.language),
        )
        if not results:
            text = ""
            language = None
        else:
            result = results[0]
            text = clean_transcript(getattr(result, "text", ""))
            language = getattr(result, "language", None)
        return Hypothesis(
            name=spec.name,
            text=text,
            backend=spec.backend,
            prior=spec.prior,
            quality=1.0,
            metadata={"language": language},
        )


class TransformersCTCBackend:
    """Local Wav2Vec2/XLS-R CTC backend with optional bundled KenLM beam search.

    The model used for the competition is explicitly trained for Indonesian,
    Javanese, and Sundanese.  Its repository contains a 5-gram language model,
    so the decoder is loaded from the bundle instead of downloading anything at
    inference time.  Greedy decoding remains a safe fallback if the optional
    ``pyctcdecode``/``kenlm`` packages are unavailable.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        use_language_model: bool = False,
        lm_alpha: float = 0.5,
        lm_beta: float = 1.5,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor
        except ImportError as exc:  # pragma: no cover - runtime-only branch
            raise RuntimeError(
                "torch and transformers are required for a CTC candidate"
            ) from exc

        self.torch = torch
        self.device = self._resolve_device(device, torch)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        try:
            self.processor = AutoProcessor.from_pretrained(
                str(model_path), local_files_only=True
            )
        except Exception:
            # Wav2Vec2ProcessorWithLM eagerly imports the native KenLM binding
            # while loading.  Build the acoustic processor directly so the
            # model still has a deterministic greedy fallback on environments
            # where that optional C++ binding is not available.
            from transformers import (
                AutoFeatureExtractor,
                AutoTokenizer,
                Wav2Vec2Processor,
            )

            self.processor = Wav2Vec2Processor(
                feature_extractor=AutoFeatureExtractor.from_pretrained(
                    str(model_path), local_files_only=True
                ),
                tokenizer=AutoTokenizer.from_pretrained(
                    str(model_path), local_files_only=True
                ),
            )
        self.model = AutoModelForCTC.from_pretrained(
            str(model_path),
            torch_dtype=dtype if self.device.type == "cuda" else None,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.decoder = None
        self.lm_enabled = False
        self.lm_alpha = float(lm_alpha)
        self.lm_beta = float(lm_beta)

        if use_language_model:
            # Wav2Vec2ProcessorWithLM already loaded the bundled KenLM graph.
            # Reuse it when available; building a second 7 GB decoder would be
            # wasteful and can cause avoidable RAM pressure on the evaluator.
            self.decoder = getattr(self.processor, "decoder", None)
            if self.decoder is None:
                self.decoder = self._try_build_decoder(model_path)
            self.lm_enabled = self.decoder is not None

    @staticmethod
    def _resolve_device(requested: str, torch_module: Any) -> Any:
        requested = str(requested).lower()
        if requested.startswith("cuda") and torch_module.cuda.is_available():
            return torch_module.device(requested)
        if requested == "mps" and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")

    def _try_build_decoder(self, model_path: Path) -> Any:
        lm_path = model_path / "language_model" / "5gram.bin"
        if not lm_path.exists():
            return None
        try:
            from pyctcdecode import build_ctcdecoder

            labels = self._labels()
            return build_ctcdecoder(labels, kenlm_model_path=str(lm_path))
        except Exception:
            # A missing optional KenLM native binding should not prevent the
            # model from running with processor/argmax decoding.
            return None

    def _labels(self) -> list[str]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        vocabulary = getattr(tokenizer, "get_vocab", lambda: {})()
        if not vocabulary:
            vocabulary = getattr(self.processor, "get_vocab", lambda: {})()
        labels: list[str] = ["<unk>"] * (max(vocabulary.values(), default=-1) + 1)
        for token, index in vocabulary.items():
            if index < 0:
                continue
            token = str(token)
            if token in {"<pad>", "[PAD]", "<s>", "</s>"}:
                token = ""
            elif token in {"|", "<unk>", "[UNK]"}:
                token = " " if token == "|" else "<unk>"
            labels[index] = token
        return labels

    @staticmethod
    def _load_audio(audio_path: Path) -> tuple[Any, int]:
        import librosa
        import soundfile as sf

        waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if getattr(waveform, "ndim", 1) > 1:
            waveform = waveform.mean(axis=1)
        if int(sample_rate) != 16000:
            waveform = librosa.resample(
                waveform.astype("float32"), orig_sr=int(sample_rate), target_sr=16000
            )
            sample_rate = 16000
        return waveform, int(sample_rate)

    def decode(self, audio_path: Path, spec: DecodeSpec) -> Hypothesis:
        import numpy as np

        waveform, sample_rate = self._load_audio(audio_path)
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            logits = self.model(**model_inputs).logits[0]
        logits_array = logits.detach().float().cpu().numpy()
        if self.decoder is not None:
            text = self.decoder.decode(
                logits_array,
                alpha=self.lm_alpha,
                beta=self.lm_beta,
            )
        else:
            ids = np.argmax(logits_array, axis=-1).tolist()
            text = self.processor.batch_decode([ids], skip_special_tokens=True)[0]
        return Hypothesis(
            name=spec.name,
            text=clean_transcript(text),
            backend=spec.backend,
            prior=spec.prior,
            quality=1.0,
            metadata={"language_model_enabled": self.lm_enabled},
        )


class EnsembleEngine:
    """Load configured local models once and transcribe clips independently."""

    def __init__(self, bundle_root: Path, config_path: Path | None = None) -> None:
        self.bundle_root = bundle_root.resolve()
        config_path = config_path or (self.bundle_root / "ensemble_config.json")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.calibration = self._load_calibration()
        # The packaged configuration targets the A100 runtime.  Environment
        # overrides make the same runner testable on a local CPU/MPS machine
        # without changing the submission configuration or model paths.
        self.device = str(os.environ.get("LIT_DEVICE", self.config.get("device", "cuda")))
        self.compute_type = str(
            os.environ.get("LIT_COMPUTE_TYPE", self.config.get("compute_type", "float16"))
        )
        self.cpu_threads = int(self.config.get("cpu_threads", 8))
        self.num_workers = int(self.config.get("num_workers", 1))
        self.specs = [
            DecodeSpec.from_mapping(value)
            for value in self.config.get("decodes", [])
            if bool(value.get("enabled", True))
        ]
        if not self.specs:
            raise ValueError("ensemble_config.json does not enable any decode candidates")
        self.extra_specs = [
            DecodeSpec.from_mapping(value)
            for value in self.config.get("adaptive_decodes", [])
            if bool(value.get("enabled", True))
        ]
        self._backends: dict[tuple[str, str, bool], Any] = {}
        self._load_backends(self.specs + self.extra_specs)

    def _load_calibration(self) -> dict[str, Any]:
        calibration_name = self.config.get("calibration_file", "calibration.json")
        calibration_path = self.bundle_root / str(calibration_name)
        if not calibration_path.exists():
            return {}
        value = json.loads(calibration_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("calibration file must contain a JSON object")
        return value

    def _model_path(self, spec: DecodeSpec) -> Path:
        path = Path(spec.model)
        if not path.is_absolute():
            path = self.bundle_root / path
        if not path.exists():
            raise FileNotFoundError(f"configured model path does not exist: {path}")
        return path

    def _load_backends(self, specs: Iterable[DecodeSpec]) -> None:
        for spec in specs:
            key = (spec.backend, spec.model, spec.use_language_model)
            if key in self._backends:
                continue
            path = self._model_path(spec)
            if spec.backend == "faster_whisper":
                self._backends[key] = FasterWhisperBackend(
                    path,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    num_workers=self.num_workers,
                )
            elif spec.backend == "qwen_asr":
                self._backends[key] = QwenASRBackend(
                    path,
                    max_new_tokens=spec.max_new_tokens,
                    device=self.device,
                )
            elif spec.backend == "transformers_ctc":
                self._backends[key] = TransformersCTCBackend(
                    path,
                    device=self.device,
                    use_language_model=spec.use_language_model,
                    lm_alpha=spec.lm_alpha,
                    lm_beta=spec.lm_beta,
                )
            else:
                raise ValueError(f"unsupported ensemble backend: {spec.backend}")

    def _decode(self, audio_path: Path, spec: DecodeSpec) -> Hypothesis:
        backend = self._backends[(spec.backend, spec.model, spec.use_language_model)]
        return backend.decode(audio_path, spec)

    def _needs_adaptive_decode(
        self, hypotheses: Sequence[Hypothesis], row: Mapping[str, Any]
    ) -> bool:
        if not self.extra_specs:
            return False
        trigger = self.config.get("adaptive_trigger", {})
        min_disagreement = float(trigger.get("min_disagreement", 0.55))
        min_duration = float(trigger.get("min_duration_s", 30.0))
        duration = 0.0
        for key in ("file_duration_seconds", "duration_s", "duration", "audio_duration_s"):
            value = _finite_or_none(row.get(key))
            if value is not None:
                duration = value
                break
        return mean_pairwise_disagreement(hypotheses) >= min_disagreement or duration >= min_duration

    def transcribe(self, audio_path: Path, row: Mapping[str, Any]) -> str:
        hypotheses = [self._decode(audio_path, spec) for spec in self.specs]
        if self._needs_adaptive_decode(hypotheses, row):
            max_extra = int(self.config.get("adaptive_trigger", {}).get("max_extra_decodes", 1))
            hypotheses.extend(self._decode(audio_path, spec) for spec in self.extra_specs[:max_extra])
        text, _, _ = select_consensus(
            hypotheses,
            calibration=self.calibration,
            language=str(row.get("language", "")) or None,
        )
        return text


def read_submission_manifest(data_dir: Path) -> list[dict[str, str]]:
    """Read the official manifest, accepting both runtime names."""

    paths = [data_dir / "submission_format.csv", data_dir / "test_metadata.csv"]
    manifest_path = next((path for path in paths if path.exists()), None)
    if manifest_path is None:
        raise FileNotFoundError("missing submission_format.csv or test_metadata.csv")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("submission manifest is empty")
    if "audio_filename" not in rows[0]:
        raise ValueError("submission manifest must contain audio_filename")
    filenames = [str(row.get("audio_filename", "")) for row in rows]
    if any(not value or Path(value).name != value or ".." in Path(value).parts for value in filenames):
        raise ValueError("manifest contains an unsafe or empty audio_filename")
    if len(set(filenames)) != len(filenames):
        raise ValueError("manifest contains duplicate audio_filename values")
    return [{str(key): str(value) for key, value in row.items()} for row in rows]


def write_submission(rows: Sequence[Mapping[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["audio_filename", "transcript"],
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "audio_filename": str(row["audio_filename"]),
                    "transcript": clean_transcript(row.get("transcript", "")),
                }
            )
