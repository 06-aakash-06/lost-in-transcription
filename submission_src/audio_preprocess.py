"""Small, deterministic audio preparation used by the offline submission.

The competition clips are supplied as compressed audio.  Each clip is decoded
independently, converted to the model sample rate, conservatively trimmed only
at the edges, and peak-safe loudness normalized.  No information from another
test clip is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _frame_rms_db(audio: np.ndarray, frame: int = 400, hop: int = 160) -> np.ndarray:
    if audio.size == 0:
        return np.array([-120.0], dtype=np.float32)
    if audio.size < frame:
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return np.array([20.0 * np.log10(max(rms, 1e-10))], dtype=np.float32)
    count = 1 + (audio.size - frame) // hop
    shape = (count, frame)
    strides = (audio.strides[0] * hop, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return (20.0 * np.log10(np.maximum(rms, 1e-10))).astype(np.float32)


def _edge_trim(audio: np.ndarray, sample_rate: int, config: Mapping[str, Any]) -> np.ndarray:
    if audio.size < 400:
        return audio
    frame = max(1, int(round(0.025 * sample_rate)))
    hop = max(1, int(round(0.010 * sample_rate)))
    levels = _frame_rms_db(audio, frame=frame, hop=hop)
    floor_db = float(np.percentile(levels, 5.0))
    threshold = floor_db + float(config.get("vad_threshold_db", 12.0))
    speech = levels > threshold
    if not speech.any():
        return audio

    indices = np.flatnonzero(speech)
    margin = int(float(config.get("edge_margin_s", 0.20)) * sample_rate)
    trim_cap = int(float(config.get("edge_trim_max_s", 1.0)) * sample_rate)
    raw_start = max(0, int(indices[0] * hop - margin))
    raw_end = min(audio.size, int(indices[-1] * hop + frame + margin))
    start = min(raw_start, trim_cap)
    end = max(raw_end, audio.size - trim_cap)
    if end - start < max(400, int(0.25 * sample_rate)):
        return audio
    return audio[start:end]


def _normalize(audio: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 1e-8:
        return audio.astype(np.float32, copy=False)
    target_db = float(config.get("target_rms_dbfs", -23.0))
    ceiling_db = float(config.get("peak_ceiling_dbfs", -1.0))
    gain = 10.0 ** ((target_db - 20.0 * np.log10(rms)) / 20.0)
    peak = float(np.max(np.abs(audio)))
    ceiling = 10.0 ** (ceiling_db / 20.0)
    if peak * gain > ceiling:
        gain = ceiling / max(peak, 1e-8)
    return (audio * gain).astype(np.float32)


def prepare_audio(
    audio_path: Path,
    config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, int]:
    """Decode one clip into a mono float32 waveform and its sample rate."""

    import librosa

    config = config or {}
    sample_rate = int(config.get("sample_rate", 16000))
    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    waveform = np.asarray(audio, dtype=np.float32)
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    if bool(config.get("edge_trim", True)):
        waveform = _edge_trim(waveform, sample_rate, config)
    if bool(config.get("loudness_normalize", True)):
        waveform = _normalize(waveform, config)
    return waveform, sample_rate
