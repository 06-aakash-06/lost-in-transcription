"""Individual, oracle, and actual deployable decode comparisons."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from final_9h_attack.compare_fusion import frame,predict,CONFIGS
from overnight_attack.evidence import score,aggregate,oracle
from phase4_stronger_anchor.audit import aligned
from phase3_protected_fusion.runtime.fusion import collapse_single_runs,repetition
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def summarize(coefficient):
    data={fold:frame(fold,coefficient) for fold in ['fold_A','fold_B']};results={}
    for label in ['indonesian_beam1','auto_beam1','indonesian_beam3']:
        record={}
        for fold,table in data.items():
            path=OUT/f'predictions/decoding/{label}/{fold}_valid.csv'
            if label=='indonesian_beam1':raw=table.large_raw.tolist()
            elif path.exists():raw=aligned(path,table)
            else:continue
            safe=list(map(collapse_single_runs,raw));changed=table.copy();changed['large_raw']=raw;changed['large']=safe
            fused=[predict(r,CONFIGS['C unambiguous boundary majority']) for r in changed.itertuples()]
            rescue=[a if a==r.turbo==r.meralion and a and repetition(a)<.2 else r.large for a,r in zip(safe,table.itertuples())]
            record[fold]={'individual':score(table.text.tolist(),safe),'C_consensus':score(table.text.tolist(),fused),'pair_oracle_vs_ID_greedy':oracle(table.text.tolist(),table.large.tolist(),safe),'exact_three_alternative_agreement_rescue':score(table.text.tolist(),rescue),'changed_greedy_clips':sum(a!=b for a,b in zip(table.large,safe))}
        if len(record)==2:record['aggregate']={k:aggregate([record[f][k] for f in data]) for k in ['individual','C_consensus','pair_oracle_vs_ID_greedy','exact_three_alternative_agreement_rescue']}
        results[label]=record
    baseline=results['indonesian_beam1'];eligible=[]
    for name,record in results.items():
        if name=='indonesian_beam1' or 'aggregate' not in record:continue
        robust=all(record[f]['C_consensus']['wer']<baseline[f]['C_consensus']['wer'] for f in data)
        material=record['aggregate']['C_consensus']['wer']<=baseline['aggregate']['C_consensus']['wer']-.0015
        record['passes_OOF_continue_gate']=robust and material
        if robust and material:eligible.append(name)
    output={'coefficient':coefficient,'results':results,'OOD_pending_candidates':eligible,'note':'Oracle is reference-selected analysis only. Whole-clip rescue requires the alternate decode, Turbo and MERaLiON to agree exactly. A global decode switch requires both-fold material consensus gains and subsequent Jember sanity.'}
    (OUT/'decoding_comparison.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--coefficient',type=float,required=True);a=p.parse_args();summarize(a.coefficient)
