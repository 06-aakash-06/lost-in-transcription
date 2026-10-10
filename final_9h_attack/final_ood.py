"""Fair, reference-blind predictions of final all-data candidates on Jember."""
import sys,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from phase4_stronger_anchor.audit import aligned
from overnight_attack.evidence import score,oracle
from final_9h_attack.runtime.consensus import consensus
from final_9h_attack.runtime.fusion import fuse,collapse_single_runs
from phase3_protected_fusion.final_conservative_check import edit_stats
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def compare():
    data=pd.read_csv(ROOT/'phase1_whisper_anchor/data/jember_holdout.tsv',sep='\t',keep_default_na=False)
    paths={
        'large':ROOT/'overnight_attack/results/predictions/final_large_v3/epoch_1_full/jember_holdout.csv',
        'large_half':OUT/'predictions/final_ood/large_half.csv',
        'large_soup075':OUT/'predictions/final_ood/large_soup075.csv',
        'turbo':OUT/'predictions/final_ood/turbo_full.csv',
        'turbo_half':OUT/'predictions/final_ood/turbo_half.csv',
        'meralion':ROOT/'phase3_protected_fusion/results/conservative_final/jember_MERaLiON_fold_A.csv'}
    for name,path in paths.items():
        if path.exists():data[name]=aligned(path,data)
    results={};predictions={}
    for name in ['large','large_half','large_soup075','turbo','turbo_half','meralion']:
        if name in data:predictions[name+'_standalone']=list(map(collapse_single_runs,data[name])) if name!='meralion' else data[name].tolist()
    if {'large','turbo','meralion'}<=set(data):
        predictions['large_consensus']=[consensus(r.large,collapse_single_runs(r.turbo),r.meralion) for r in data.itertuples()]
    if {'large_soup075','turbo','meralion'}<=set(data):
        predictions['large_soup075_consensus']=[consensus(r.large_soup075,collapse_single_runs(r.turbo),r.meralion) for r in data.itertuples()]
    if {'turbo','turbo_half','meralion'}<=set(data):
        with zipfile.ZipFile(ROOT/'artifacts/FINAL_SUBMISSION_CONSERVATIVE.zip') as z:cfg=json.loads(z.read('config.json'))['fusion']
        predictions['old_conservative']=[fuse(r.turbo,r.meralion,cfg,third=r.turbo_half) for r in data.itertuples()]
    for name,pred in predictions.items():
        anchor=name.removesuffix('_standalone') if name.endswith('_standalone') else 'turbo' if name=='old_conservative' else 'large_soup075' if 'soup075' in name else 'large'
        anchors=data[anchor].tolist() if name.endswith('_standalone') else list(map(collapse_single_runs,data[anchor]))
        results[name]=score(data.text.tolist(),pred)
        stats=[edit_stats(a,p) for a,p in zip(anchors,pred)]
        results[name]['edits']={k:sum(e[k] for e in stats) for k in stats[0]}
        results[name]['changed_clips']=sum(a!=p for a,p in zip(anchors,pred))
        results[name]['average_corrections_per_clip']=results[name]['edits']['word_edits']/len(data)
        path=OUT/f'predictions/final_ood/combined_{name}.csv';path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'clip_id':data.clip_id,'reference':data.text,'transcript':pred}).to_csv(path,index=False)
    oracles={}
    for name,columns in [('Large+Turbo',['large','turbo']),('Large+MER',['large','meralion']),('triple',['large','turbo','meralion'])]:
        if set(columns)<=set(data):oracles[name]=oracle(data.text.tolist(),*[data[c].map(collapse_single_runs).tolist() if c!='meralion' else data[c].tolist() for c in columns])
    output={'dataset':'same untouched 162 Jember holdout clips, 6689 words','models':'Final all-data checkpoints, not fold checkpoints','results':results,'oracles':oracles,'oracle_note':'Reference-selected diagnostic only, never deployed.'}
    (OUT/'final_ood_comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    for name,result in results.items():print(name,result['wer'],result['edits'],flush=True)
    return output

if __name__=='__main__':compare()
