"""Offline ASR ensemble runtime used by the competition submission.

The module is intentionally dependency-light at import time.  The optional ASR
packages are imported only when their configured backend is instantiated.  This
keeps the local evaluation and consensus code testable on a CPU-only machine
while the packaged submission uses the competition runtime's GPU dependencies.

The ensemble is transcript-level rather than logit-level because the candidate
models use different tokenizers.  The default weighted medoid is a deliberately
conservative minimum-risk selector.  A calibrated word-level ROVER path is also
available for the mixed-language route where held-out validation showed that
aligning the strongest candidates can improve on selecting one whole transcript.
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


# Backends accept either a path or an already decoded 16 kHz waveform.  The
# latter lets the engine share deterministic preprocessing across candidates.
AudioSource = Path | tuple[Any, int]


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


def canonical_manifest_language(value: Any) -> str:
    """Map manifest spellings to the competition's three language labels."""

    raw = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "id": "ind",
        "ind": "ind",
        "indonesian": "ind",
        "jav": "jav",
        "jv": "jav",
        "jw": "jav",
        "javanese": "jav",
        "javind": "javind",
        "ind-jav": "javind",
        "jav-ind": "javind",
        "id-jv": "javind",
        "jv-id": "javind",
        "id-jav": "javind",
        "jav-id": "javind",
    }
    return aliases.get(raw, raw)


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
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.0
    chunk_length_s: float = 30.0
    chunk_overlap_s: float = 1.0
    long_form: bool = False
    padding_mode: str = "dynamic"
    use_language_model: bool = False
    lm_alpha: float = 0.5
    lm_beta: float = 1.5
    prior: float = 1.0
    enabled: bool = True
    manifest_languages: tuple[str, ...] = ()
    preprocess_profile: str | None = None

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
            no_repeat_ngram_size=int(value.get("no_repeat_ngram_size", 0)),
            repetition_penalty=float(value.get("repetition_penalty", 1.0)),
            chunk_length_s=float(value.get("chunk_length_s", 30.0)),
            chunk_overlap_s=float(value.get("chunk_overlap_s", 1.0)),
            long_form=bool(value.get("long_form", False)),
            padding_mode=str(value.get("padding_mode", "dynamic")),
            use_language_model=bool(value.get("use_language_model", False)),
            lm_alpha=float(value.get("lm_alpha", 0.5)),
            lm_beta=float(value.get("lm_beta", 1.5)),
            prior=float(value.get("prior", 1.0)),
            enabled=bool(value.get("enabled", True)),
            manifest_languages=tuple(str(item) for item in value.get("manifest_languages", ())),
            preprocess_profile=(
                None
                if value.get("preprocess_profile") in (None, "")
                else str(value.get("preprocess_profile"))
            ),
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


def reference_closeness_index(
    hypotheses: Sequence[Hypothesis],
    calibration: Mapping[str, Any],
) -> int | None:
    """Select a specialist close to a trusted anchor, with a safe fallback.

    The anchor is the strongest general model on the target-like validation
    slice.  A specialist is promoted only when its normalized word distance to
    that anchor is below the calibrated guard.  This avoids selecting a
    plausible-looking but catastrophically repeated or empty decode merely
    because it is one of several candidates.
    """

    settings = calibration.get("reference_closeness", {})
    if not isinstance(settings, Mapping):
        return None
    anchor_name = str(settings.get("anchor", ""))
    anchor_index = next(
        (index for index, hypothesis in enumerate(hypotheses) if hypothesis.name == anchor_name),
        None,
    )
    if anchor_index is None:
        return None
    candidate_names = settings.get("candidate_names", ())
    if not isinstance(candidate_names, (list, tuple, set)):
        candidate_names = ()
    candidate_name_set = {str(value) for value in candidate_names}
    candidates = [
        index
        for index, hypothesis in enumerate(hypotheses)
        if index != anchor_index
        and (not candidate_name_set or hypothesis.name in candidate_name_set)
    ]
    if not candidates:
        return anchor_index
    try:
        max_distance = float(settings.get("max_normalized_disagreement", 0.6))
    except (TypeError, ValueError):
        max_distance = 0.6
    max_distance = max(0.0, min(2.0, max_distance))
    anchor_tokens = hypotheses[anchor_index].tokens
    best_index = min(
        candidates,
        key=lambda index: (
            normalized_distance(anchor_tokens, hypotheses[index].tokens),
            -float(hypotheses[index].prior),
            index,
        ),
    )
    if normalized_distance(anchor_tokens, hypotheses[best_index].tokens) <= max_distance:
        return best_index
    return anchor_index


def _align_token_sequences(
    left: Sequence[str], right: Sequence[str]
) -> list[tuple[str | None, str | None]]:
    """Return a deterministic minimum-edit alignment of two token sequences."""

    rows = len(left) + 1
    columns = len(right) + 1
    costs = [[0] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
    for column in range(1, columns):
        costs[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            substitution = costs[row - 1][column - 1] + (left[row - 1] != right[column - 1])
            deletion = costs[row - 1][column] + 1
            insertion = costs[row][column - 1] + 1
            costs[row][column] = min(substitution, deletion, insertion)

    aligned: list[tuple[str | None, str | None]] = []
    row, column = len(left), len(right)
    while row or column:
        # Prefer diagonal moves on ties, then deletion, then insertion.  This
        # keeps repeated words and local insertions stable across runs.
        if row and column:
            substitution_cost = costs[row - 1][column - 1] + (
                left[row - 1] != right[column - 1]
            )
            if costs[row][column] == substitution_cost:
                aligned.append((left[row - 1], right[column - 1]))
                row -= 1
                column -= 1
                continue
        if row and costs[row][column] == costs[row - 1][column] + 1:
            aligned.append((left[row - 1], None))
            row -= 1
            continue
        if column and costs[row][column] == costs[row][column - 1] + 1:
            aligned.append((None, right[column - 1]))
            column -= 1
            continue
        raise RuntimeError("failed to backtrace token alignment")
    aligned.reverse()
    return aligned


def _rover_representative(
    column: Mapping[str, str | None],
    weights: Mapping[str, float],
    order_rank: Mapping[str, int],
) -> str | None:
    """Choose the weighted majority token used to align the next hypothesis."""

    token_scores: dict[str, float] = {}
    for name, token in column.items():
        if token is not None:
            token_scores[token] = token_scores.get(token, 0.0) + weights.get(name, 1.0)
    if not token_scores:
        return None
    return min(
        token_scores,
        key=lambda token: (
            -token_scores[token],
            min(
                order_rank.get(name, len(order_rank))
                for name, value in column.items()
                if value == token
            ),
            token,
        ),
    )


def rover_consensus(
    hypotheses: Sequence[Hypothesis],
    weights: Sequence[float],
    settings: Mapping[str, Any],
) -> str | None:
    """Fuse selected hypotheses with weighted word-level ROVER voting.

    ``blank_factor`` controls conservatism: a word must receive at least that
    multiple of the aligned blank weight to survive.  The function returns
    ``None`` only when there are too few usable candidates; an empty string is
    a valid consensus for an all-empty set of hypotheses.
    """

    if len(hypotheses) != len(weights):
        raise ValueError("hypotheses and weights must have the same length")
    if not hypotheses:
        return None
    candidate_names = settings.get("candidate_names", ())
    if isinstance(candidate_names, (list, tuple, set)) and candidate_names:
        allowed = {str(value) for value in candidate_names}
        selected = [
            (hypothesis, float(weight))
            for hypothesis, weight in zip(hypotheses, weights)
            if hypothesis.name in allowed
        ]
    else:
        selected = [(hypothesis, float(weight)) for hypothesis, weight in zip(hypotheses, weights)]
    if not selected:
        return None

    selected_by_name = {hypothesis.name: (hypothesis, weight) for hypothesis, weight in selected}
    configured_order = settings.get("order", ())
    order = [
        str(name)
        for name in configured_order
        if str(name) in selected_by_name
    ] if isinstance(configured_order, (list, tuple)) else []
    order.extend(name for name in selected_by_name if name not in order)
    if len(order) < 2:
        return selected_by_name[order[0]][0].cleaned_text if order else None

    weights_by_name = {name: selected_by_name[name][1] for name in order}
    order_rank = {name: index for index, name in enumerate(order)}
    first_name = order[0]
    columns: list[dict[str, str | None]] = [
        {first_name: token} for token in selected_by_name[first_name][0].tokens
    ]
    for name in order[1:]:
        tokens = selected_by_name[name][0].tokens
        representatives = [
            _rover_representative(column, weights_by_name, order_rank) or ""
            for column in columns
        ]
        aligned = _align_token_sequences(representatives, tokens)
        next_columns: list[dict[str, str | None]] = []
        left_index = 0
        for left_token, right_token in aligned:
            if left_token is not None:
                column = dict(columns[left_index])
                left_index += 1
            else:
                column = {}
            if right_token is not None:
                column[name] = right_token
            next_columns.append(column)
        if left_index != len(columns):
            raise RuntimeError("ROVER alignment did not consume all existing columns")
        columns = next_columns

    try:
        blank_factor = float(settings.get("blank_factor", 1.25))
    except (TypeError, ValueError):
        blank_factor = 1.25
    blank_factor = max(0.0, min(10.0, blank_factor))
    output: list[str] = []
    for column in columns:
        token_scores: dict[str, float] = {}
        blank_score = 0.0
        for name in order:
            token = column.get(name)
            if token is None:
                blank_score += weights_by_name[name]
            else:
                token_scores[token] = token_scores.get(token, 0.0) + weights_by_name[name]
        if not token_scores:
            continue
        token = min(
            token_scores,
            key=lambda value: (
                -token_scores[value],
                min(order_rank[name] for name in order if column.get(name) == value),
                value,
            ),
        )
        if token_scores[token] >= blank_factor * blank_score:
            output.append(token)
    return clean_transcript(" ".join(output))


def duration_override_candidate(
    calibration: Mapping[str, Any], language: str | None, duration_s: float | None
) -> str | None:
    """Return a calibrated candidate override for a duration/language rule."""

    if not language or duration_s is None:
        return None
    try:
        duration = float(duration_s)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration):
        return None
    rules = calibration.get("duration_overrides", ())
    if not isinstance(rules, (list, tuple)):
        return None
    for rule in rules:
        if not isinstance(rule, Mapping):
            continue
        languages = rule.get("languages", ())
        if isinstance(languages, str):
            languages = (languages,)
        if languages and str(language) not in {str(value) for value in languages}:
            continue
        try:
            minimum = float(rule.get("min_duration_s", float("-inf")))
            maximum = float(rule.get("max_duration_s", float("inf")))
        except (TypeError, ValueError):
            continue
        if minimum <= duration < maximum and rule.get("candidate_name"):
            return str(rule["candidate_name"])
    return None


def row_duration_seconds(row: Mapping[str, Any]) -> float | None:
    """Read the first usable duration field from a runtime manifest row."""

    for key in ("file_duration_seconds", "duration_s", "duration", "audio_duration_s"):
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


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
    duration_s: float | None = None,
) -> tuple[str, str, float]:
    """Return ``(raw_text, selected_name, disagreement)``."""

    if not hypotheses:
        raise ValueError("No ASR hypotheses were generated")
    weights = candidate_weights(hypotheses, calibration=calibration, language=language)
    calibration = calibration or {}
    selection = str(calibration.get("selection", "weighted_medoid"))
    selection_by_language = calibration.get("selection_by_language", {})
    if language and isinstance(selection_by_language, Mapping):
        selection = str(selection_by_language.get(language, selection))
    override_name = duration_override_candidate(calibration, language, duration_s)
    if override_name:
        override_index = next(
            (index for index, hypothesis in enumerate(hypotheses) if hypothesis.name == override_name),
            None,
        )
        if override_index is not None:
            disagreement = mean_pairwise_disagreement(hypotheses)
            selected = hypotheses[override_index]
            return selected.cleaned_text, selected.name, disagreement
    if selection == "rover":
        settings = calibration.get("rover", {})
        rover_by_language = calibration.get("rover_by_language", {})
        if language and isinstance(rover_by_language, Mapping):
            language_settings = rover_by_language.get(language)
            if isinstance(language_settings, Mapping):
                settings = language_settings
        if isinstance(settings, Mapping):
            fused = rover_consensus(hypotheses, weights, settings)
            if fused is not None:
                disagreement = mean_pairwise_disagreement(hypotheses)
                return fused, "rover", disagreement
        selection = "weighted_medoid"
    if selection == "reference_closeness":
        index = reference_closeness_index(hypotheses, calibration)
        if index is None:
            index = weighted_medoid_index(hypotheses, weights)
    else:
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

    def decode(self, audio_source: AudioSource, spec: DecodeSpec) -> Hypothesis:
        audio_input: str | Any
        if isinstance(audio_source, Path):
            audio_input = str(audio_source)
        else:
            audio_input = audio_source[0]
        segments, info = self.model.transcribe(
            audio_input,
            language=spec.language,
            task="transcribe",
            beam_size=spec.beam_size,
            repetition_penalty=max(0.1, float(spec.repetition_penalty)),
            no_repeat_ngram_size=max(0, int(spec.no_repeat_ngram_size)),
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
        max_inference_batch_size: int = 8,
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
            max_inference_batch_size=max(1, int(max_inference_batch_size)),
            max_new_tokens=max_new_tokens,
        )

    @staticmethod
    def _audio_input(audio_source: AudioSource) -> str | tuple[Any, int]:
        if isinstance(audio_source, Path):
            return str(audio_source)
        return audio_source

    @staticmethod
    def _hypothesis(result: Any, spec: DecodeSpec) -> Hypothesis:
        if result is None:
            text = ""
            language = None
        else:
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

    def decode(self, audio_source: AudioSource, spec: DecodeSpec) -> Hypothesis:
        results = self.model.transcribe(
            audio=self._audio_input(audio_source),
            language=_qwen_language(spec.language),
        )
        return self._hypothesis(results[0] if results else None, spec)

    def decode_many(
        self, audio_sources: Sequence[AudioSource], specs: Sequence[DecodeSpec]
    ) -> list[Hypothesis]:
        if not audio_sources:
            return []
        if len(audio_sources) != len(specs):
            raise ValueError("audio_sources and specs must have the same length")
        results = self.model.transcribe(
            audio=[self._audio_input(source) for source in audio_sources],
            language=[_qwen_language(spec.language) for spec in specs],
        )
        if len(results) != len(specs):
            raise RuntimeError("Qwen returned an unexpected number of transcripts")
        return [self._hypothesis(result, spec) for result, spec in zip(results, specs)]


class MERaLiONASRBackend:
    """Local wrapper for the MERaLiON-3 Southeast Asian ASR checkpoint.

    MERaLiON is a custom Transformers model whose chat prompt is part of the
    inference contract.  It is decoded in small GPU batches so the 3B text
    decoder does not turn a multi-thousand-clip test set into thousands of
    sequential model launches.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        max_batch_size: int = 4,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:  # pragma: no cover - runtime-only branch
            raise RuntimeError(
                "torch and transformers are required for a MERaLiON candidate"
            ) from exc

        self.torch = torch
        self.max_batch_size = max(1, int(max_batch_size))
        requested = str(device).lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(requested)
            self.model_dtype = torch.bfloat16
        elif requested == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.model_dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            self.model_dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=True
        )
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
            "dtype": self.model_dtype,
            # SDPA is substantially faster on the competition A100.  Eager
            # attention remains the portable fallback for MPS/CPU smoke tests.
            "attn_implementation": "sdpa" if self.device.type == "cuda" else "eager",
        }
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(model_path), **load_kwargs
        )
        self.model.to(self.device)
        self.model.eval()
        self.prompt = self.processor.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "Instruction: Please transcribe this speech. \n"
                        "Follow the text instruction based on the following audio: "
                        "<SpeechHere>"
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _load_audio(audio_source: AudioSource) -> tuple[Any, int]:
        if not isinstance(audio_source, Path):
            return audio_source
        import librosa

        waveform, sample_rate = librosa.load(
            str(audio_source), sr=16000, mono=True
        )
        return waveform, int(sample_rate)

    def _strip_chat_wrapper(self, text: Any) -> str:
        value = str(text or "")
        # MERaLiON's decoder returns the chat prompt together with the newly
        # generated answer when using batch_decode.  Keep only the assistant
        # turn; this also makes the output safe for the official scorer.
        for marker in ("\nmodel\n", "model\n"):
            if marker in value:
                value = value.rsplit(marker, 1)[-1]
                break
        return clean_transcript(value)

    def _move_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        model_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(self.device)
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    value = value.to(dtype=self.model_dtype)
            model_inputs[key] = value
        return model_inputs

    def _decode_batch(
        self, audio_sources: Sequence[AudioSource], specs: Sequence[DecodeSpec]
    ) -> list[Hypothesis]:
        audios = []
        sample_rate = 16000
        for audio_source in audio_sources:
            audio, sample_rate = self._load_audio(audio_source)
            audios.append(audio)
        inputs = self.processor(
            text=[self.prompt for _ in audios],
            audios=audios,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = self._move_inputs(inputs)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max(1, max(int(spec.max_new_tokens) for spec in specs)),
            "do_sample": False,
            "no_repeat_ngram_size": max(
                0, max(int(spec.no_repeat_ngram_size) for spec in specs)
            ),
        }
        repetition_penalty = max(float(spec.repetition_penalty) for spec in specs)
        if repetition_penalty != 1.0:
            generation_kwargs["repetition_penalty"] = max(
                0.1, repetition_penalty
            )
        with self.torch.inference_mode():
            generated = self.model.generate(**model_inputs, **generation_kwargs)
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [
            Hypothesis(
                name=spec.name,
                text=self._strip_chat_wrapper(text),
                backend=spec.backend,
                prior=spec.prior,
                quality=1.0,
                metadata={"language": None, "chat_wrapper_stripped": True},
            )
            for spec, text in zip(specs, decoded)
        ]

    def decode_many(
        self, audio_sources: Sequence[AudioSource], specs: Sequence[DecodeSpec]
    ) -> list[Hypothesis]:
        if not audio_sources:
            return []
        if len(audio_sources) != len(specs):
            raise ValueError("audio_sources and specs must have the same length")
        outputs: list[Hypothesis] = []
        for start in range(0, len(audio_sources), self.max_batch_size):
            end = start + self.max_batch_size
            outputs.extend(self._decode_batch(audio_sources[start:end], specs[start:end]))
        return outputs

    def decode(self, audio_source: AudioSource, spec: DecodeSpec) -> Hypothesis:
        return self.decode_many([audio_source], [spec])[0]


class TransformersWhisperBackend:
    """Batched Hugging Face Whisper backend for a specialist checkpoint.

    This is kept separate from the faster-whisper wrapper because a fine-tuned
    Whisper checkpoint may only be available in Transformers format.  The
    backend accepts the same already-prepared waveform representation as the
    other candidates and keeps generation fully local.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:  # pragma: no cover - runtime-only branch
            raise RuntimeError(
                "torch and transformers are required for a Transformers Whisper candidate"
            ) from exc

        self.torch = torch
        requested = str(device).lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(requested)
            self.model_dtype = torch.float16
        elif requested == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.model_dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            self.model_dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        load_kwargs: dict[str, Any] = {"local_files_only": True}
        if self.device.type != "cpu":
            load_kwargs["torch_dtype"] = self.model_dtype
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(model_path), **load_kwargs
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_audio(audio_source: AudioSource) -> tuple[Any, int]:
        if not isinstance(audio_source, Path):
            return audio_source
        import librosa

        waveform, sample_rate = librosa.load(str(audio_source), sr=16000, mono=True)
        return waveform, int(sample_rate)

    def _forced_decoder_ids(self, language: str | None) -> Any:
        if not language:
            return None
        language = self._language_code(language)
        getter = getattr(self.processor, "get_decoder_prompt_ids", None)
        if getter is None:
            getter = getattr(getattr(self.processor, "tokenizer", None), "get_decoder_prompt_ids", None)
        if getter is None:
            return None
        try:
            return getter(language=language, task="transcribe")
        except (KeyError, ValueError):
            return None

    @staticmethod
    def _language_code(language: str) -> str:
        language = str(language).lower()
        aliases = {
            "ind": "id",
            "indonesian": "id",
            "jav": "jw",
            "javind": "jw",
            "javanese": "jw",
        }
        return aliases.get(language, language)

    def decode_many(
        self, audio_sources: Sequence[AudioSource], specs: Sequence[DecodeSpec]
    ) -> list[Hypothesis]:
        if not audio_sources:
            return []
        if len(audio_sources) != len(specs):
            raise ValueError("audio_sources and specs must have the same length")

        loaded = [self._load_audio(source) for source in audio_sources]
        short_indices: list[int] = []
        long_indices: list[int] = []
        for index, ((audio, sample_rate), spec) in enumerate(zip(loaded, specs)):
            limit = max(5.0, float(spec.chunk_length_s)) * int(sample_rate)
            if spec.long_form and len(audio) > limit:
                long_indices.append(index)
            else:
                short_indices.append(index)

        results: list[Hypothesis | None] = [None] * len(audio_sources)
        if short_indices:
            short_specs = [specs[index] for index in short_indices]
            short_audios = [loaded[index][0] for index in short_indices]
            decoded = self._decode_waveform_batch(short_audios, short_specs)
            for index, hypothesis in zip(short_indices, decoded):
                results[index] = hypothesis
        for index in long_indices:
            results[index] = self._decode_long_form(loaded[index], specs[index])
        if any(result is None for result in results):
            raise RuntimeError("Whisper did not produce one result per audio source")
        return [result for result in results if result is not None]

    def _decode_waveform_batch(
        self, audios: Sequence[Any], specs: Sequence[DecodeSpec]
    ) -> list[Hypothesis]:
        """Decode already-loaded clips that fit in Whisper's 30-second window."""

        languages = {spec.language for spec in specs}
        if len(languages) > 1:
            return [
                self._decode_waveform_batch([audio], [spec])[0]
                for audio, spec in zip(audios, specs)
            ]

        padding_mode = "max_length" if any(
            str(spec.padding_mode).lower() in {"max", "max_length"} for spec in specs
        ) else True
        inputs = self.processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            # Dynamic padding is the default.  A candidate may request the
            # processor's original fixed 30-second padding when that decoding
            # profile is calibrated separately.
            padding=padding_mode,
            truncation=True,
            return_attention_mask=True,
        )
        model_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(self.device)
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    value = value.to(dtype=self.model_dtype)
            model_inputs[key] = value

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max(1, max(spec.max_new_tokens for spec in specs)),
            "num_beams": max(1, max(spec.beam_size for spec in specs)),
            "do_sample": False,
            "condition_on_prev_tokens": False,
        }
        no_repeat_ngram_size = max(spec.no_repeat_ngram_size for spec in specs)
        if no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        repetition_penalty = max(spec.repetition_penalty for spec in specs)
        if repetition_penalty != 1.0:
            generation_kwargs["repetition_penalty"] = repetition_penalty
        language = specs[0].language
        if language:
            generation_kwargs["language"] = self._language_code(language)
            generation_kwargs["task"] = "transcribe"

        with self.torch.inference_mode():
            generated = self.model.generate(**model_inputs, **generation_kwargs)
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [
            Hypothesis(
                name=spec.name,
                text=clean_transcript(text),
                backend=spec.backend,
                prior=spec.prior,
                quality=1.0,
                metadata={"language": spec.language},
            )
            for spec, text in zip(specs, texts)
        ]

    @staticmethod
    def _merge_chunk_texts(texts: Sequence[str]) -> str:
        """Concatenate overlapping chunk decodes without duplicating words."""

        merged: list[str] = []
        for text in texts:
            current = clean_transcript(text).split()
            if not current:
                continue
            if not merged:
                merged = current
                continue
            max_overlap = min(20, len(merged), len(current))
            overlap = 0
            for size in range(max_overlap, 0, -1):
                left = alignment_tokens(" ".join(merged[-size:]))
                right = alignment_tokens(" ".join(current[:size]))
                if left == right:
                    overlap = size
                    break
            merged.extend(current[overlap:])
        return clean_transcript(" ".join(merged))

    def _decode_long_form(
        self, loaded: tuple[Any, int], spec: DecodeSpec
    ) -> Hypothesis:
        """Decode clips longer than 30 seconds with overlapping windows.

        The previous implementation silently truncated these clips at the
        processor's 30-second feature limit.  The competition allows clips up
        to about 40 seconds, so preserving the tail is more important than
        saving one extra model call.
        """

        audio, sample_rate = loaded
        chunk_samples = max(1, int(float(spec.chunk_length_s) * sample_rate))
        overlap_samples = max(0, int(float(spec.chunk_overlap_s) * sample_rate))
        if overlap_samples >= chunk_samples:
            overlap_samples = max(0, chunk_samples // 20)
        step = max(1, chunk_samples - overlap_samples)
        starts = [0]
        while starts[-1] + chunk_samples < len(audio):
            candidate = starts[-1] + step
            remaining = len(audio) - (starts[-1] + chunk_samples)
            # A tiny tail is not worth a second almost-identical decode.  The
            # processor's 30-second window loses at most this short margin.
            if remaining <= overlap_samples:
                break
            if candidate + chunk_samples > len(audio):
                candidate = len(audio) - chunk_samples
            if candidate <= starts[-1]:
                break
            starts.append(candidate)
        chunks = [audio[start : start + chunk_samples] for start in starts]
        chunk_specs = [spec for _ in chunks]
        chunk_hypotheses = self._decode_waveform_batch(chunks, chunk_specs)
        text = self._merge_chunk_texts([item.text for item in chunk_hypotheses])
        return Hypothesis(
            name=spec.name,
            text=text,
            backend=spec.backend,
            prior=spec.prior,
            quality=min((item.quality for item in chunk_hypotheses), default=1.0),
            metadata={
                "language": spec.language,
                "long_form": True,
                "n_chunks": len(chunks),
            },
        )

    def decode(self, audio_source: AudioSource, spec: DecodeSpec) -> Hypothesis:
        return self.decode_many([audio_source], [spec])[0]


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
        self.model_dtype = dtype
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
            return build_ctcdecoder(
                labels,
                kenlm_model_path=str(lm_path),
                alpha=self.lm_alpha,
                beta=self.lm_beta,
            )
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
    def _load_audio(audio_source: AudioSource) -> tuple[Any, int]:
        if not isinstance(audio_source, Path):
            return audio_source
        import librosa
        import soundfile as sf

        waveform, sample_rate = sf.read(str(audio_source), dtype="float32", always_2d=False)
        if getattr(waveform, "ndim", 1) > 1:
            waveform = waveform.mean(axis=1)
        if int(sample_rate) != 16000:
            waveform = librosa.resample(
                waveform.astype("float32"), orig_sr=int(sample_rate), target_sr=16000
            )
            sample_rate = 16000
        return waveform, int(sample_rate)

    def decode(self, audio_source: AudioSource, spec: DecodeSpec) -> Hypothesis:
        import numpy as np

        waveform, sample_rate = self._load_audio(audio_source)
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(self.device)
                # The processor emits floating-point audio features as FP32,
                # while the CUDA checkpoint is loaded in FP16.  Cast only
                # floating tensors; attention masks and other integer inputs
                # must retain their original dtype.
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    value = value.to(dtype=self.model_dtype)
            model_inputs[key] = value
        with self.torch.inference_mode():
            logits = self.model(**model_inputs).logits[0]
        logits_array = logits.detach().float().cpu().numpy()
        if self.decoder is not None:
            # BeamSearchDecoderCTC stores alpha/beta in the decoder.  Passing
            # them here is incompatible with the official pyctcdecode API.
            text = self.decoder.decode(logits_array)
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
        self.audio_preprocessing = dict(self.config.get("audio_preprocessing", {}))
        self.qwen_batch_size = max(1, int(self.config.get("qwen_batch_size", 8)))
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
                    max_inference_batch_size=self.qwen_batch_size,
                )
            elif spec.backend == "meralion_asr":
                self._backends[key] = MERaLiONASRBackend(
                    path,
                    device=self.device,
                    max_batch_size=int(self.config.get("meralion_batch_size", 4)),
                )
            elif spec.backend == "transformers_whisper":
                self._backends[key] = TransformersWhisperBackend(
                    path,
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

    def _decode(self, audio_source: AudioSource, spec: DecodeSpec) -> Hypothesis:
        backend = self._backends[(spec.backend, spec.model, spec.use_language_model)]
        return backend.decode(audio_source, spec)

    def _decode_many(
        self,
        audio_sources: Sequence[AudioSource],
        specs: Sequence[DecodeSpec],
    ) -> list[Hypothesis]:
        if not audio_sources:
            return []
        if len(audio_sources) != len(specs):
            raise ValueError("audio_sources and specs must have the same length")
        backend = self._backends[(specs[0].backend, specs[0].model, specs[0].use_language_model)]
        if (
            all(
                (spec.backend, spec.model, spec.use_language_model)
                == (specs[0].backend, specs[0].model, specs[0].use_language_model)
                for spec in specs
            )
            and hasattr(backend, "decode_many")
        ):
            return backend.decode_many(audio_sources, specs)
        return [self._decode(audio, spec) for audio, spec in zip(audio_sources, specs)]

    @staticmethod
    def _spec_matches_language(spec: DecodeSpec, manifest_language: str) -> bool:
        return not spec.manifest_languages or manifest_language in {
            canonical_manifest_language(value) for value in spec.manifest_languages
        }

    def _route_language(self, row: Mapping[str, Any]) -> str:
        # In this competition the test language can identify the track, not
        # the clip.  An explicit track route prevents leaked dev labels from
        # controlling decoder selection or fusion calibration.
        configured = self.config.get("route_language")
        return canonical_manifest_language(
            configured if configured is not None else row.get("language", "")
        )

    def _active_specs(self, row: Mapping[str, Any]) -> list[DecodeSpec]:
        manifest_language = self._route_language(row)
        routed = [
            spec
            for spec in self.specs
            if self._spec_matches_language(spec, manifest_language)
        ]
        # Keep a permissive fallback only for missing or future labels.  A
        # known monolingual row must not silently activate an unrelated forced
        # language candidate because a route was omitted from the config.
        if routed:
            return routed
        if manifest_language in {"ind", "jav", "javind"}:
            raise RuntimeError(f"no configured decode candidate for language={manifest_language}")
        return list(self.specs)

    def _prepare_audio(
        self, audio_path: Path, spec: DecodeSpec | None = None
    ) -> AudioSource:
        profile_name = str(
            spec.preprocess_profile
            if spec is not None and spec.preprocess_profile
            else self.audio_preprocessing.get("default_profile", "default")
        )
        profiles = self.audio_preprocessing.get("profiles", {})
        if isinstance(profiles, Mapping):
            profile = profiles.get(profile_name, profiles.get("default", {}))
        else:
            profile = self.audio_preprocessing
        if not isinstance(profile, Mapping) or not bool(profile.get("enabled", False)):
            return audio_path
        from audio_preprocess import prepare_audio

        return prepare_audio(audio_path, profile)

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
        active_specs = self._active_specs(row)
        hypotheses = [
            self._decode(self._prepare_audio(audio_path, spec), spec)
            for spec in active_specs
        ]
        if self._needs_adaptive_decode(hypotheses, row):
            max_extra = int(self.config.get("adaptive_trigger", {}).get("max_extra_decodes", 1))
            hypotheses.extend(
                self._decode(self._prepare_audio(audio_path, spec), spec)
                for spec in self.extra_specs[:max_extra]
                if self._spec_matches_language(spec, self._route_language(row))
            )
        text, _, _ = select_consensus(
            hypotheses,
            calibration=self.calibration,
            language=self._route_language(row) or None,
            duration_s=row_duration_seconds(row),
        )
        return text

    def transcribe_batch(
        self, audio_paths: Sequence[Path], rows: Sequence[Mapping[str, Any]]
    ) -> list[str]:
        """Transcribe independent clips, batching only Qwen model calls."""

        if len(audio_paths) != len(rows):
            raise ValueError("audio_paths and rows must have the same length")
        if not rows:
            return []
        hypotheses_by_row: list[list[Hypothesis]] = [[] for _ in rows]
        active_specs = [self._active_specs(row) for row in rows]
        source_cache: dict[tuple[int, str], AudioSource] = {}

        def source_for(index: int, spec: DecodeSpec) -> AudioSource:
            profile_name = str(
                spec.preprocess_profile
                if spec.preprocess_profile
                else self.audio_preprocessing.get("default_profile", "default")
            )
            key = (index, profile_name)
            if key not in source_cache:
                source_cache[key] = self._prepare_audio(audio_paths[index], spec)
            return source_cache[key]

        for spec in self.specs:
            selected = [
                index for index, specs in enumerate(active_specs) if spec in specs
            ]
            if not selected:
                continue
            backend = self._backends[(spec.backend, spec.model, spec.use_language_model)]
            if hasattr(backend, "decode_many"):
                batch_sources = [source_for(index, spec) for index in selected]
                batch_specs = [spec for _ in selected]
                decoded = self._decode_many(batch_sources, batch_specs)
                for index, hypothesis in zip(selected, decoded):
                    hypotheses_by_row[index].append(hypothesis)
            else:
                for index in selected:
                    hypotheses_by_row[index].append(
                        self._decode(source_for(index, spec), spec)
                    )

        # Adaptive candidates are decoded in backend batches too.  Calling an
        # extra specialist once per row is needlessly slow on the execution
        # platform, especially for a candidate such as the Javanese Whisper
        # model that supports batched generation.
        adaptive_triggered = [
            index
            for index, row in enumerate(rows)
            if self._needs_adaptive_decode(hypotheses_by_row[index], row)
        ]
        max_extra = int(self.config.get("adaptive_trigger", {}).get("max_extra_decodes", 1))
        adaptive_batch_size = max(1, int(self.config.get("inference_batch_size", 8)))
        for spec in self.extra_specs[:max_extra]:
            selected = [
                index
                for index in adaptive_triggered
                if self._spec_matches_language(
                    spec, self._route_language(rows[index])
                )
            ]
            backend = self._backends[(spec.backend, spec.model, spec.use_language_model)]
            for start in range(0, len(selected), adaptive_batch_size):
                batch_indices = selected[start : start + adaptive_batch_size]
                if not batch_indices:
                    continue
                batch_sources = [source_for(index, spec) for index in batch_indices]
                batch_specs = [spec for _ in batch_indices]
                decoded = self._decode_many(batch_sources, batch_specs)
                for index, hypothesis in zip(batch_indices, decoded):
                    hypotheses_by_row[index].append(hypothesis)

        outputs: list[str] = []
        for index, row in enumerate(rows):
            hypotheses = hypotheses_by_row[index]
            if not hypotheses:
                raise RuntimeError("no decode candidate was active for a manifest row")
            outputs.append(
                select_consensus(
                    hypotheses,
                    calibration=self.calibration,
                    language=self._route_language(row) or None,
                    duration_s=row_duration_seconds(row),
                )[0]
            )
        return outputs


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
