"""The single optional experiment: cautious half/full sequence support."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from final_9h_attack.compare_fusion import frame,consensus,unique_map
from final_9h_attack.runtime.consensus import proposals
from overnight_attack.evidence import score,aggregate,oracle
from phase4_stronger_anchor.audit import aligned
from phase3_protected_fusion.runtime.fusion import collapse_single_runs
from phase3_protected_fusion.final_conservative_check import edit_stats
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def candidate(anchor,turbo,mer,half):
    baseline=consensus(anchor,turbo,mer,0).split();original=anchor.split()
    # Context-qualified half+MER support only; preserve every existing C edit.
    for i,replacement in proposals(anchor,half,mer,1):
        if baseline[i]==original[i]:baseline[i]=replacement
    return ' '.join(baseline)

def evaluate():
    results={};oracles={}
    for dataset in ['dev','jember']:
        metrics={};base={};ores={}
        for fold in ['fold_A','fold_B']:
            data=frame(fold,1.,dataset)
            half=frame(fold,0.,dataset).large.tolist()
            pred=[candidate(r.large,r.turbo,r.meralion,h) for r,h in zip(data.itertuples(),half)]
            old=[consensus(r.large,r.turbo,r.meralion,0) for r in data.itertuples()]
            metrics[fold]=score(data.text.tolist(),pred);base[fold]=score(data.text.tolist(),old)
            edits=[edit_stats(a,b) for a,b in zip(old,pred)]
            metrics[fold]['additional_edits']={k:sum(e[k] for e in edits) for k in edits[0]}
            ores[fold]={'half_full':oracle(data.text.tolist(),data.large.tolist(),half),'half_full_mer':oracle(data.text.tolist(),data.large.tolist(),half,data.meralion.tolist())}
        metrics['aggregate']=aggregate([metrics[f] for f in base]);base['aggregate']=aggregate(list(base.values()))
        results[dataset]={'candidate':metrics,'baseline_C':base};oracles[dataset]=ores
    dev=results['dev'];j=results['jember']
    passed=all(dev['candidate'][f]['wer']<dev['baseline_C'][f]['wer'] for f in ['fold_A','fold_B']) and dev['candidate']['aggregate']['wer']<=dev['baseline_C']['aggregate']['wer']-.002 and j['candidate']['aggregate']['wer']<=j['baseline_C']['aggregate']['wer']+.001
    output={'experiment':'One context-qualified half/full sequence support rule; preserve core C corrections','results':results,'oracles':oracles,'passes_continue_gate':passed,'keep_policy':'Requires both-fold gains, >=.002 aggregate improvement and Jember worsening <=.001 to justify another full decoder pass. No training, broad threshold search, or reference features.'}
    (OUT/'checkpoint_ensemble_comparison.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':evaluate()
