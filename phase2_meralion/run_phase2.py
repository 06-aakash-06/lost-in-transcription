"""Gated sequential MPS execution. No submission builder, no Phase 1 writes."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase2_meralion.analyze import analyze,aggregate
from phase2_meralion.report import report
HERE=Path(__file__).resolve().parent

def command(script,*args):
    print('RUN',script,*map(str,args),flush=True)
    subprocess.run([sys.executable,str(HERE/script),*map(str,args)],check=True)

def train(name,manifest,checkpoint="epoch_1_full"):
    run=HERE/'runs'/name
    if not (run/checkpoint/'adapter_config.json').exists():
        args=['--manifest',ROOT/f'phase1_whisper_anchor/data/{manifest}.tsv','--output',run]
        if checkpoint=='epoch_1_half':args+=['--stop-after-half']
        command('training/train_lora.py',*args)
    return run

def evaluate_fold(fold,run):
    result={}
    # Evaluate full immediately, then half; only two checkpoints.
    for checkpoint in ('epoch_1_full','epoch_1_half'):
        if not (run/checkpoint).exists():continue
        path=HERE/f'results/pred_{checkpoint}_{fold}.csv'
        command('evaluate.py','--manifest',ROOT/f'phase1_whisper_anchor/data/{fold}_valid.tsv','--adapter',run/checkpoint,'--output',path)
        result[checkpoint]=analyze(fold,path,HERE/f'results/adapted_{checkpoint}_{fold}.json')
    return result

def main(recover_half=False):
    gate=HERE/'results/zero_shot_aggregate.json'

    deadline=time.monotonic()+7200
    while not gate.exists():
        if time.monotonic()>deadline:raise RuntimeError('fresh zero-shot gate did not finish within two hours')
        time.sleep(10)
    time.sleep(5)  # let the independent zero-shot process release its MPS model
    if not json.loads(gate.read_text())['passes_training_gate']:
        report('ABANDON MERALION','Fresh zero-shot oracle gain did not pass the 0.01 gate.');return
    if recover_half:
        # Keep the finite, saved checkpoint after the interrupted full epoch.
        # Wait for its already-running evaluation rather than duplicate it.
        while not (HERE/'results/pred_epoch_1_half_fold_A.json').exists():time.sleep(10)
    checkpoint='epoch_1_half' if recover_half else 'epoch_1_full'
    a=evaluate_fold('fold_A',train('fold_A','fold_A_train',checkpoint))
    best_a=min(a,key=lambda k:(a[k]['oracle']['wer'],a[k]['meralion']['wer']))
    if a[best_a]['oracle_gain']<.01:
        report('ABANDON MERALION','Adapted Fold A oracle gain is below 0.01; Fold B and final training are skipped.',best_a);return
    b=evaluate_fold('fold_B',train('fold_B','fold_B_train',checkpoint))
    both={key:aggregate([a[key],b[key]]) for key in a.keys()&b.keys()}
    selected=min(both,key=lambda key:(max(a[key]['oracle']['wer'],b[key]['oracle']['wer']),both[key]['oracle']['wer'],both[key]['meralion']['wer']))
    summary=both[selected];summary.update({'selected_checkpoint':selected,'passes_final_gate':summary['oracle_gain']>=.01 and a[selected]['oracle_gain']>=.01})
    (HERE/'results/adapted_aggregate.json').write_text(json.dumps(summary,indent=2)+'\n')
    (HERE/'results/checkpoint_comparison.json').write_text(json.dumps(both,indent=2)+'\n')
    if not summary['passes_final_gate']:
        report('ABANDON MERALION','Adapted pair failed the final complementarity gate; no final MERaLiON model is trained.',selected);return
    report('IN PROGRESS','Both adaptation gates pass. Final all-data training is running.',selected)
    final=train('final','final_train',selected)
    command('training/merge_lora.py','--adapter',final/selected,'--output',HERE/'models/final_merged')
    # Load the actual saved standalone checkpoint offline, retaining no PEFT dependency.
    command('evaluate.py','--base',HERE/'models/final_merged','--manifest',ROOT/'phase1_whisper_anchor/data/fold_A_valid.tsv','--output',HERE/'results/final_merged_smoke.csv','--limit',2)
    report('PROCEED TO PHASE 3','Adapted pair oracle passes the complementarity gate; two-fold OOF predictions and a verified standalone model are ready.',selected)
    print('PHASE 2 COMPLETE',summary,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--recover-half',action='store_true',help='Use the finite Fold A half checkpoint and matched half-epoch B/final runs after the interrupted full epoch.')
    main(parser.parse_args().recover_half)
