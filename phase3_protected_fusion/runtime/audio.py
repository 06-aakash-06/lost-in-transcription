"""Decode to mono float32 at 16 kHz and construct overlapping windows."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

RATE = 16000

def load_audio(path: str | Path, redact_errors: bool = False) -> np.ndarray:
    try:
        wave, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        if redact_errors: raise RuntimeError("audio decoding failed") from None
        raise
    if wave.size == 0: raise ValueError("empty audio")
    mono = wave.mean(axis=1)
    if sr != RATE: mono = librosa.resample(mono, orig_sr=sr, target_sr=RATE)
    return np.ascontiguousarray(mono, dtype=np.float32)

def windows(wave: np.ndarray, window_s: float = 27.0, overlap_s: float = 2.0) -> list[np.ndarray]:
    if wave.ndim != 1 or not len(wave): raise ValueError("audio must be nonempty mono")
    width, overlap = round(window_s*RATE), round(overlap_s*RATE)
    if width > 30*RATE or not 0 < overlap < width: raise ValueError("invalid windows")
    if len(wave) <= width: return [wave]
    starts = list(range(0, len(wave), width-overlap))
    return [wave[start:start+width] for start in starts]
