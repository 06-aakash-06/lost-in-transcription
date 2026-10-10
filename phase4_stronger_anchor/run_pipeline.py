"""Sequential GPU jobs, cached evaluation, explicit 21-hour wall-clock budget."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'phase4_stronger_anchor'
RESULTS = HERE / 'results'
CHECKPOINTS = ['epoch_1_half', 'epoch_1_full', 'epoch_2_half', 'epoch_2_full']

def record(stage, **extra):
    data = {'stage': stage, 'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat(), **extra}
    path = RESULTS / 'pipeline_status.json'
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(data, indent=2)+'\n'); temp.replace(path)
    print(json.dumps(data), flush=True)

def call(name, *args):
    record('running', job=name, command=list(map(str, args)))
    with (RESULTS / f'{name}.log').open('a') as log:
        subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    record('job_complete', job=name)

def eval_checkpoint(run, checkpoint, manifest_name, language='indonesian'):
    checkpoint_dir = HERE / 'runs' / run / checkpoint if run == 'final_indonesian_x4_two_epochs' or run.startswith('indonesian_x4_two_epochs_') else ROOT / 'phase1_whisper_anchor/runs' / run / checkpoint
    output = RESULTS / 'predictions' / run / checkpoint / f'{manifest_name}_{language}.json'
    # Evaluator verifies cache provenance even when complete; it performs no
    # additional model load/decoding for existing complete predictions.
    call(f'eval_{run}_{checkpoint}_{manifest_name}_{language}', HERE / 'evaluate.py', '--manifest',
        ROOT / f'phase1_whisper_anchor/data/{manifest_name}.tsv', '--checkpoint', checkpoint_dir,
        '--output', output, '--language', language, '--batch-size', '4')

def main():
    p = argparse.ArgumentParser(); p.add_argument('--initial-training-pid', type=int, required=True)
    p.add_argument('--start-utc', required=True); args = p.parse_args()
    start = dt.datetime.fromisoformat(args.start_utc).timestamp()
    deadline = start + 21*3600
    record('waiting_final_ratio4', deadline_utc=dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat())
    final_run = 'final_indonesian_x4_two_epochs'
    log = RESULTS / 'final_ratio4_training.log'
    while 'TRAINING COMPLETE' not in log.read_text():
        if 'Traceback (most recent call last)' in log.read_text(): raise RuntimeError('Final Ratio-4 training failed; inspect its log.')
        try: os.kill(args.initial_training_pid, 0)
        except ProcessLookupError: raise RuntimeError('Initial training exited without completion marker.')
        time.sleep(20)
    # Highest-priority all-data run is fully complete before any fold job.
    for checkpoint in CHECKPOINTS:
        eval_checkpoint(final_run, checkpoint, 'jember_holdout')
    # Honest A/B evidence needs the same two-epoch schedule with each complete
    # competition conversation withheld. Cached one-epoch runs are retained.
    for fold in ['fold_A', 'fold_B']:
        if time.time() > deadline - 5*3600: raise RuntimeError('Insufficient budget for an honest OOF run and final packaging.')
        run = f'indonesian_x4_two_epochs_{fold}'
        folder = HERE / 'runs' / run
        if not (folder / 'epoch_2_full/training_state.pt').exists():
            existing = [folder / c for c in CHECKPOINTS if (folder / c / 'training_state.pt').exists()]
            command = [HERE / 'train.py', '--config', HERE / 'configs/ratio4_two_epochs.json', '--manifest',
                ROOT / f'phase1_whisper_anchor/data/{fold}_train.tsv', '--output', folder]
            if existing: command += ['--resume', existing[-1]]
            call(f'train_{run}', *command)
        for checkpoint in CHECKPOINTS:
            for manifest in [f'{fold}_valid', 'jember_holdout']:
                eval_checkpoint(run, checkpoint, manifest)
    call('select_checkpoints', HERE / 'compare.py', '--step', 'checkpoints')
    selected = json.loads((RESULTS / 'checkpoint_selection.json').read_text())['selected_checkpoint']
    record('decode_mode_experiment', checkpoint=selected)
    for fold in ['fold_A', 'fold_B']:
        run = f'indonesian_x4_two_epochs_{fold}'
        for language in ['javanese', 'auto']:
            for manifest in [f'{fold}_valid', 'jember_holdout']:
                eval_checkpoint(run, selected, manifest, language)
    for language in ['javanese', 'auto']:
        eval_checkpoint(final_run, selected, 'jember_holdout', language)
    # Old final weights on this holdout have no saved predictions, so this
    # comparison is useful new evidence, rather than repeated dev inference.
    for checkpoint in ['epoch_1_half', 'epoch_1_full']:
        eval_checkpoint('final_indonesian_x3', checkpoint, 'jember_holdout')
    call('compare_decode_fusion', HERE / 'compare.py', '--step', 'fusion')
    record('evaluation_complete', checkpoint=selected, remaining_hours=(deadline-time.time())/3600)

if __name__ == '__main__':
    try: main()
    except Exception as exc:
        record('failed', error=repr(exc)); raise
