"""Start immediately after bounded checkpoint selection and validate decoding."""
import argparse,os,json,time,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from final_9h_attack.jobs import run,EXPERIMENT_END
from overnight_attack.soup import average
from final_9h_attack.prepare_model import prepare
from final_9h_attack.decode_summary import summarize
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def select():
    records=json.loads((OUT/'checkpoint_comparison.json').read_text())
    complete={float(k):v for k,v in records.items() if all(n in v for n in ['aggregate','jember','worst_fold'])}
    full=complete[1.]
    eligible={k:v for k,v in complete.items() if all(v[f]['wer']<=full[f]['wer']+.003 for f in ['fold_A','fold_B']) and v['worst_fold']<=full['worst_fold']+.003}
    best=min(eligible,key=lambda k:eligible[k]['aggregate']['wer'])
    near={k:v for k,v in eligible.items() if v['aggregate']['wer']<=eligible[best]['aggregate']['wer']+.002}
    ood=min(near,key=lambda k:near[k]['jember']['wer'])
    # Prefer pure full when it is near the OOF leader and not meaningfully
    # worse on Jember. Otherwise require a real Jember gain to trade OOF.
    chosen=1. if 1. in near and near[1.]['jember']['wer']<=near[ood]['jember']['wer']+.001 else ood if near[best]['jember']['wer']-near[ood]['jember']['wer']>=.002 else best
    selected={'coefficient':chosen,'language':'indonesian','beams':1,'prediction_mode':'adapted','runtime_mode':'exact_delta_pending_proof','selection_metrics':complete[chosen],
        'policy':'Both folds within .003 of full; near-best OOF within .002; favor stable Jember and pure full simplicity. A soup must provide >=.002 Jember gain to trade away the OOF-leading score, unless full is already near-best and no worse on Jember.'}
    (OUT/'anchor_selection.json').write_text(json.dumps(selected,indent=2)+'\n');print('SELECTED ANCHOR',json.dumps(selected),flush=True)
    return selected

def execute(wait_pid):
    while wait_pid:
        state=subprocess.run(['ps','-o','stat=','-p',str(wait_pid)],capture_output=True,text=True).stdout.strip()
        if not state or state.startswith('Z'):break
        time.sleep(2)
    selected=select();coefficient=selected['coefficient'];start=time.time();deadline=min(start+3600,EXPERIMENT_END)
    final=ROOT/'overnight_attack/runs/final_large_v3';adapter=final/'epoch_1_full' if coefficient==1 else final/'epoch_1_half' if coefficient==0 else HERE/f'models/final_soup_{coefficient}'
    if coefficient not in (0,1):average(final/'epoch_1_half',final/'epoch_1_full',adapter,1-coefficient)
    model=HERE/'models/selected_large';prepare(adapter,model)
    if not run([HERE/'prove_delta.py','--adapter',adapter,'--output',OUT/'delta_runtime_equivalence.json'],'prove_selected_delta',deadline):raise RuntimeError('Delta equivalence needs repair before packaging')
    selected.update(runtime_mode='exact_delta_verified',model_path=str(model.resolve()),adapter_path=str(adapter.resolve()))
    (OUT/'anchor_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    run([HERE/'decode_variants.py','--coefficient',str(coefficient),'--deadline',str(deadline)],'decode_variants_controller',deadline)
    summarize(coefficient)
    (OUT/'phase2_done.json').write_text(json.dumps({'completed_at':time.time(),'phase_deadline':deadline,'coefficient':coefficient},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);a=p.parse_args();execute(a.wait_pid)
