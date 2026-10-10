"""Offline, restartable raw decoding with cached provenance and safe metrics."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase1_whisper_anchor.inference.whisper import WhisperAnchor
from phase3_protected_fusion.runtime.whisper import WhisperAnchor as RawWhisper
from phase3_protected_fusion.runtime.fusion import collapse_single_runs
from phase1_whisper_anchor.inference.audio import load_audio
from phase1_whisper_anchor.training.metrics import corpus

def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)

def evaluate(manifest, checkpoint, output, language='indonesian', batch_size=4):
    frame = pd.read_csv(manifest, sep='\t', keep_default_na=False)
    assert frame.clip_id.is_unique
    base = ROOT / 'submission_src/models/whisper_large_v3_turbo'
    adapter_file = checkpoint / 'adapter_model.safetensors'
    with adapter_file.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    provenance = {'checkpoint': str(checkpoint.resolve()), 'adapter_sha256': digest,
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'language': language,
        'decode': 'greedy_raw_native_long_form_440_tokens', 'batch_size': batch_size}
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = output.with_suffix('.cache.json')
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {'provenance': provenance, 'predictions': {}, 'elapsed_s': 0.}
    assert cache['provenance'] == provenance, ('cache provenance mismatch', cache_path)
    predictions = cache['predictions']
    assert set(predictions) <= set(frame.clip_id)
    for row in frame.itertuples():
        if row.clip_id in predictions: assert predictions[row.clip_id]['reference'] == row.text
    remaining = frame[~frame.clip_id.isin(predictions)]
    if len(remaining):
        torch.manual_seed(1337)
        engine = WhisperAnchor.from_adapter(str(base), checkpoint, None if language == 'auto' else language, batch_size, 'native')
        # Phase 1 currently applies a repetition collapse inside its decoder.
        # Preserve raw text here so raw/safe comparisons use one shared rule.
        engine._decode = RawWhisper._decode.__get__(engine, WhisperAnchor)
        engine.model.generation_config.language = None
        started = time.monotonic(); prior_elapsed = cache['elapsed_s']
        for start in range(0, len(remaining), batch_size):
            part = remaining.iloc[start:start+batch_size]
            texts = engine.transcribe_arrays([load_audio(ROOT / path) for path in part.path])
            assert len(texts) == len(part)
            for row, text in zip(part.itertuples(), texts):
                if not isinstance(text, str): raise TypeError('nontext decode')
                predictions[row.clip_id] = {'reference': row.text, 'transcript': text}
            cache['elapsed_s'] = prior_elapsed + time.monotonic() - started
            write(cache_path, cache)
            if (start // batch_size) % 10 == 0:
                print(f'{output.stem} decoded={len(predictions)}/{len(frame)} elapsed_s={cache["elapsed_s"]:.1f}', flush=True)
        del engine
    texts = [predictions[clip]['transcript'] for clip in frame.clip_id]
    safe = list(map(collapse_single_runs, texts))
    stats = {'provenance': provenance, 'raw': corpus(frame.text.tolist(), texts), 'safe': corpus(frame.text.tolist(), safe),
        'repeat_changed_clips': sum(a != b for a, b in zip(texts, safe)), 'clips': len(frame), 'elapsed_s': cache['elapsed_s'],
        'leakage_note': 'Competition OOF only for a matching fold-trained checkpoint; all-data final models may only use Jember holdout diagnostics.'}
    pd.DataFrame({'clip_id': frame.clip_id, 'reference': frame.text, 'transcript': texts}).to_csv(output.with_suffix('.csv'), index=False)
    write(output, stats)
    print('EVALUATION COMPLETE', output, json.dumps({'raw': stats['raw']['wer'], 'safe': stats['safe']['wer']}), flush=True)
    return stats

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--language', choices=['indonesian', 'javanese', 'auto'], default='indonesian')
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args(); evaluate(a.manifest, a.checkpoint, a.output, a.language, a.batch_size)
