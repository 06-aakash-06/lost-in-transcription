"""Check a fixed catastrophic phrase guard across every cached candidate."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from final_9h_attack.compare_fusion import frame,consensus
from final_9h_attack.runtime.repetition_guard import guard
from final_9h_attack.runtime.fusion import collapse_single_runs
from overnight_attack.evidence import score,aggregate,error
from phase3_protected_fusion.final_conservative_check import edit_stats
from final_9h_attack.final_ood import compare as final_compare
import pandas as pd
from phase4_stronger_anchor.audit import aligned
OUT=Path(__file__).resolve().parent/'results'

def guarded_consensus(raw,turbo,mer):
    safe=guard(raw)
    return safe if safe!=raw else consensus(raw,turbo,mer,0)

def compare():
    candidates={}
    for coefficient in [0,.25,.5,.75,1]:
        results={}
        for dataset in ['dev','jember']:
            rows={}
            for fold in ['fold_A','fold_B']:
                data=frame(fold,coefficient,dataset);pred=[guarded_consensus(r.large_raw,r.turbo,r.meralion) for r in data.itertuples()]
                safe=list(map(guard,data.large_raw));old=data.large.tolist()
                row={'standalone':score(data.text.tolist(),safe),'consensus':score(data.text.tolist(),pred),'changed_vs_single_guard':sum(a!=b for a,b in zip(old,safe)),'guard_helped':sum(sum(error(r,a)[1:])>sum(error(r,b)[1:]) for r,a,b in zip(data.text,old,safe)),'guard_harmed':sum(sum(error(r,a)[1:])<sum(error(r,b)[1:]) for r,a,b in zip(data.text,old,safe))}
                for label,left,right in [('fusion_edits',safe,pred),('guard_edits',data.large_raw,safe)]:
                    edits=[edit_stats(a,b) for a,b in zip(left,right)];row[label]={k:sum(e[k] for e in edits) for k in edits[0]}
                rows[fold]=row
                p=OUT/f'phrase_guard/{coefficient}/{dataset}_{fold}.csv';p.parent.mkdir(parents=True,exist_ok=True)
                pd.DataFrame({'clip_id':data.clip_id,'reference':data.text,'transcript':pred}).to_csv(p,index=False)
            rows['aggregate']={k:aggregate([rows[f][k] for f in ['fold_A','fold_B']]) for k in ['standalone','consensus']};results[dataset]=rows
        candidates[str(coefficient)]=results
    final=final_compare();data=pd.read_csv(ROOT/'phase1_whisper_anchor/data/jember_holdout.tsv',sep='\t',keep_default_na=False)
    turbo=aligned(OUT/'predictions/final_ood/turbo_full.csv',data);mer=aligned(ROOT/'phase3_protected_fusion/results/conservative_final/jember_MERaLiON_fold_A.csv',data)
    diagnostics={}
    for name,path in [('full',ROOT/'overnight_attack/results/predictions/final_large_v3/epoch_1_full/jember_holdout.csv'),('half',OUT/'predictions/final_ood/large_half.csv'),('soup075',OUT/'predictions/final_ood/large_soup075.csv')]:
        raw=aligned(path,data);safe=list(map(guard,raw));pred=[guarded_consensus(a,collapse_single_runs(t),m) for a,t,m in zip(raw,turbo,mer)]
        diagnostics[name]={'standalone':score(data.text.tolist(),safe),'consensus':score(data.text.tolist(),pred),'guard_changed':sum(a!=b for a,b in zip(raw,safe)),'changed_vs_single_guard':sum(collapse_single_runs(a)!=b for a,b in zip(raw,safe))}
        for label,left,right in [('fusion_edits',safe,pred),('guard_edits',raw,safe)]:
            edits=[edit_stats(a,b) for a,b in zip(left,right)];diagnostics[name][label]={k:sum(e[k] for e in edits) for k in edits[0]}
        pd.DataFrame({'clip_id':data.clip_id,'reference':data.text,'transcript':pred}).to_csv(OUT/f'phrase_guard/final_{name}.csv',index=False)
    output={'guard':'Existing single-word guard plus phrases of 2-6 words repeated >=5 times, >=20 loop words and >=35% of output, retaining two phrase cycles. No further voting after cleanup. One fixed rule; no identity/ref features.','candidates':candidates,'final_ood':diagnostics}
    (OUT/'phrase_guard_comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    print('FINAL GUARDED',json.dumps(diagnostics),flush=True)
    for coefficient,r in candidates.items():print(coefficient,'OOF',r['dev']['aggregate']['consensus']['wer'],'J',r['jember']['aggregate']['consensus']['wer'],'changed/harmed',[(r[d][f]['changed_vs_single_guard'],r[d][f]['guard_harmed']) for d in ['dev','jember'] for f in ['fold_A','fold_B']],flush=True)
if __name__=='__main__':compare()
