"""Recover every requested Phase 1 result from cached predictions, no decode."""
import json
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase1_whisper_anchor.training.metrics import corpus
from phase3_protected_fusion.runtime.fusion import collapse_single_runs
OUT = ROOT / 'phase4_stronger_anchor/results'

def aligned(path, manifest):
    cached = pd.read_csv(path, keep_default_na=False).set_index('clip_id')
    assert cached.index.is_unique and set(cached.index) == set(manifest.clip_id), path
    cached = cached.loc[manifest.clip_id]
    assert cached.reference.tolist() == manifest.text.tolist(), path
    return cached.transcript.tolist()

def main():
    rows = []
    evidence = {}
    runs = ROOT / 'phase1_whisper_anchor/runs'
    for language, ratio in [('indonesian', 3), ('indonesian', 2), ('indonesian', 4), ('auto', 3), ('javanese', 3)]:
        for checkpoint in ['epoch_1_half', 'epoch_1_full']:
            refs_all = []; raw_all = []; safe_all = []; fold_stats = {}; holdout_stats = {}
            for fold in ['fold_A', 'fold_B']:
                directory = runs / f'{language}_x{ratio}_{fold}' / checkpoint
                manifest = pd.read_csv(ROOT / f'phase1_whisper_anchor/data/{fold}_valid.tsv', sep='\t', keep_default_na=False)
                raw = aligned(directory / 'valid.csv', manifest)
                safe = list(map(collapse_single_runs, raw))
                fold_stats[fold] = {'raw': corpus(manifest.text.tolist(), raw), 'safe': corpus(manifest.text.tolist(), safe),
                    'repeat_changed_clips': sum(a != b for a, b in zip(raw, safe))}
                holdout = pd.read_csv(ROOT / 'phase1_whisper_anchor/data/jember_holdout.tsv', sep='\t', keep_default_na=False)
                held_raw = aligned(directory / 'jember.csv', holdout)
                holdout_stats[fold] = {'raw': corpus(holdout.text.tolist(), held_raw), 'safe': corpus(holdout.text.tolist(), list(map(collapse_single_runs, held_raw)))}
                old = json.loads((directory / 'valid.json').read_text())
                assert abs(old['wer'] - fold_stats[fold]['raw']['wer']) < 1e-10, ('saved metrics drift', directory)
                refs_all += manifest.text.tolist(); raw_all += raw; safe_all += safe
            name = f'{language}_x{ratio}_{checkpoint}'
            evidence[name] = {'folds': fold_stats, 'jember': holdout_stats, 'aggregate_raw': corpus(refs_all, raw_all), 'aggregate_safe': corpus(refs_all, safe_all)}
            rows.append({'run': name, 'A_raw': fold_stats['fold_A']['raw']['wer'], 'B_raw': fold_stats['fold_B']['raw']['wer'],
                'aggregate_raw': evidence[name]['aggregate_raw']['wer'], 'A_safe': fold_stats['fold_A']['safe']['wer'],
                'B_safe': fold_stats['fold_B']['safe']['wer'], 'aggregate_safe': evidence[name]['aggregate_safe']['wer'],
                'Jember_A_raw': holdout_stats['fold_A']['raw']['wer'], 'Jember_B_raw': holdout_stats['fold_B']['raw']['wer'],
                'Jember_mean_raw': sum(holdout_stats[f]['raw']['wer'] for f in holdout_stats)/2,
                'Jember_mean_safe': sum(holdout_stats[f]['safe']['wer'] for f in holdout_stats)/2})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'existing_evidence.csv', index=False)
    (OUT / 'existing_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    table = ['| Run | A raw | B raw | Aggregate raw | A safe | B safe | Aggregate safe | Jember raw | Jember safe |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        keys = ['A_raw', 'B_raw', 'aggregate_raw', 'A_safe', 'B_safe', 'aggregate_safe', 'Jember_mean_raw', 'Jember_mean_safe']
        table.append('| ' + r['run'] + ' | ' + ' | '.join(f'{r[k]:.6f}' for k in keys) + ' |')
    text = '# Existing evidence audit\n\nRecovered from all 40 saved valid/Jember prediction CSVs, aligned against the manifests. No inference rerun. Safe collapses only five or more identical consecutive words.\n\n' + '\n'.join(table)
    text += '\n\nJember columns average the two fold models; they are diagnostics, not final-model holdout measurements. All-data final models have seen both competition conversations and must never be reported as OOF.\n'
    text += '\nOld R3 checkpoints contain adapter weights and train metrics only. Exact optimizer/scheduler continuation is unavailable. A weights-only restart would be a distinct experiment.\n'
    (OUT / 'EXISTING_EVIDENCE.md').write_text(text)
    print('\n'.join(table))

if __name__ == '__main__': main()
