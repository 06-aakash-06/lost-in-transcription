"""One merged Whisper model, greedy decode, explicit overlapping long windows."""
from __future__ import annotations
import time
import os
from pathlib import Path
import numpy as np
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from .audio import RATE, windows
from .postprocess import clean, collapse_catastrophic, merge_overlap

class WhisperAnchor:
    def __init__(self, model_dir: str | Path, language: str | None = "indonesian", batch_size: int = 8, long_mode: str = "native"):
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir(): raise FileNotFoundError("merged model directory missing")
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if self.device=="mps" and hasattr(torch.mps,"set_per_process_memory_fraction"):
            try: torch.mps.set_per_process_memory_fraction(0.9)
            except (RuntimeError,NotImplementedError): pass
        self.dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.processor = WhisperProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_dir, local_files_only=True, torch_dtype=self.dtype).to(self.device).eval()
        self.language = language
        self.batch_size = batch_size
        self.long_mode = long_mode

    @torch.inference_mode()
    def _decode(self, waves: list[np.ndarray], long_form: bool = False) -> list[str]:
        if not waves: return []
        if long_form:
            if len(waves) != 1: raise ValueError("native long-form requires one clip")
            features = self.processor.feature_extractor(waves[0], sampling_rate=RATE, truncation=False, padding="longest", return_tensors="pt", return_attention_mask=True)
        else:
            features = self.processor.feature_extractor(waves, sampling_rate=RATE, return_tensors="pt", return_attention_mask=True)
        kwargs = {"num_beams": 1, "do_sample": False, "task": "transcribe", "return_timestamps": long_form}
        if self.language is not None: kwargs["language"] = self.language
        if long_form: kwargs["max_new_tokens"] = 440
        else: kwargs["max_new_tokens"] = 440
        ids = self.model.generate(input_features=features.input_features.to(self.device, dtype=self.dtype), attention_mask=features.attention_mask.to(self.device), **kwargs)
        # Fusion applies its validated repetition rule after confidence mapping.
        return [clean(s) for s in self.processor.batch_decode(ids, skip_special_tokens=True)]

    def _decode_safe(self, waves: list[np.ndarray]) -> list[str]:
        try:
            return self._decode(waves)
        except RuntimeError as exc:
            message=str(exc).lower()
            is_mps_memory_error=self.device=="mps" and len(waves)>1 and ("out of memory" in message or "mps backend out of memory" in message)
            if not is_mps_memory_error: raise
            torch.mps.empty_cache()
            middle=len(waves)//2
            return self._decode_safe(waves[:middle])+self._decode_safe(waves[middle:])

    def transcribe_arrays(self, waves: list[np.ndarray]) -> list[str]:
        results = [""] * len(waves)
        safe_limit = (30 if self.long_mode == "native" else 27) * RATE
        short_indices = [i for i,w in enumerate(waves) if len(w) <= safe_limit]
        for start in range(0,len(short_indices),self.batch_size):
            ids = short_indices[start:start+self.batch_size]
            batch = self._decode_safe([waves[i] for i in ids])
            for i,s in zip(ids,batch): results[i] = s
        for i,wave in enumerate(waves):
            if i in short_indices: continue
            if self.long_mode == "native": results[i] = self._decode([wave], long_form=True)[0]
            elif self.long_mode == "overlap":
                chunks = windows(wave)
                parts = []
                for start in range(0,len(chunks),self.batch_size): parts.extend(self._decode_safe(chunks[start:start+self.batch_size]))
                results[i] = merge_overlap(parts)
            else: raise ValueError("long_mode must be native or overlap")
        return results
