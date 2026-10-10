"""Fixed, metadata-blind Large-v3 consensus comparisons from aligned caches."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from phase4_stronger_anchor.audit import aligned
from overnight_attack.evidence import score,aggregate,oracle
from phase3_protected_fusion.runtime.fusion import fuse,collapse_single_runs,alignment,key,blocks
from phase3_protected_fusion.final_conservative_check import edit_stats
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def large_path(fold,coefficient,dataset):
    selection=OUT/'anchor_selection.json'
    if selection.exists():
        selected=json.loads(selection.read_text())
        if selected['coefficient']==coefficient and selected.get('prediction_mode')=='merged':
            return OUT/f'predictions/merged/{coefficient}/{fold}/{dataset}.csv'
    if coefficient in (0,1):
        checkpoint='epoch_1_half' if coefficient==0 else 'epoch_1_full'
        return ROOT/f'overnight_attack/results/predictions/large_v3_{fold}/{checkpoint}/{dataset}.csv'
    return OUT/f'predictions/soup_{coefficient}/{fold}/{dataset}.csv'

def frame(fold,coefficient,dataset='dev'):
    name=f'{fold}_valid' if dataset=='dev' else 'jember_holdout'
    data=pd.read_csv(ROOT/f'phase1_whisper_anchor/data/{name}.tsv',sep='\t',keep_default_na=False)
    data['large_raw']=aligned(large_path(fold,coefficient,name),data)
    data['large']=data.large_raw.map(collapse_single_runs)
    for model,checkpoint in [('turbo','epoch_1_full'),('half','epoch_1_half')]:
        data[model]=list(map(collapse_single_runs,aligned(ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/{checkpoint}/{"valid" if dataset=="dev" else "jember"}.csv',data)))
    mer=ROOT/f'phase2_meralion/results/pred_epoch_1_half_{fold}.csv' if dataset=='dev' else ROOT/'phase3_protected_fusion/results/conservative_final/jember_MERaLiON_fold_A.csv'
    data['meralion']=aligned(mer,data)
    old=ROOT/f'phase3_protected_fusion/results/conservative_final/{dataset if dataset=="dev" else "jember"}_B_{fold}.csv'
    data['old']=aligned(old,data)
    return data

def unique_map(a,b):
    """Only diagonals shared by every minimum-cost alignment are usable."""
    x,y=list(map(key,a)),list(map(key,b));n,m=len(x),len(y)
    forward=[[0]*(m+1) for _ in range(n+1)];reverse=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):forward[i][0]=i;reverse[i][m]=n-i
    for j in range(m+1):forward[0][j]=j;reverse[n][j]=m-j
    for i in range(1,n+1):
        for j in range(1,m+1):forward[i][j]=min(forward[i-1][j-1]+(x[i-1]!=y[j-1]),forward[i-1][j]+1,forward[i][j-1]+1)
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):reverse[i][j]=min(reverse[i+1][j+1]+(x[i]!=y[j]),reverse[i+1][j]+1,reverse[i][j+1]+1)
    optimum=forward[n][m];mapping={}
    for i in range(n):
        if any(forward[i][j]+1+reverse[i+1][j]==optimum for j in range(m+1)):continue
        js=[j for j in range(m) if forward[i][j]+(x[i]!=y[j])+reverse[i+1][j+1]==optimum]
        if len(js)==1:mapping[i]=js[0]
    return mapping

def consensus(anchor,first,second,context=1):
    a,b,c=anchor.split(),first.split(),second.split();mb,mc=unique_map(a,b),unique_map(a,c)
    result=a.copy()
    for i in range(len(a)):
        if i not in mb or i not in mc:continue
        candidate=b[mb[i]]
        if candidate!=c[mc[i]] or key(candidate)==key(a[i]):continue
        if context and (i==0 or i==len(a)-1):continue
        if context and any(j not in mb or j not in mc or key(a[j])!=key(b[mb[j]]) or key(a[j])!=key(c[mc[j]]) for j in [i-1,i+1]):continue
        result[i]=candidate
    return ' '.join(result)

BASE={'mode':'support','max_span':3,'context':1,'collapse_anchor':True,'operations':['substitution'],'strict_substitutions':True}
CONFIGS={
    'Large standalone':None,
    'Old conservative':None,
    'A Turbo+MER strict':{'strategy':'A','fusion':BASE},
    'A Turbo+MER short indels':{'strategy':'A','fusion':{**BASE,'max_span':1,'operations':['substitution','insertion','deletion'],'strict_substitutions':False}},
    'B Turbo half+full strict':{'strategy':'B','fusion':BASE},
    'C unambiguous word majority':{'strategy':'C','context':1},
    'C unambiguous boundary majority':{'strategy':'C','context':0},
}

def predict(row,config,confidence=None):
    if row.large_raw!=row.large:return row.large
    if config['strategy']=='C':return consensus(row.large,row.turbo,row.meralion,config['context'])
    first,second=(row.meralion,row.turbo) if config['strategy']=='A' else (row.turbo,row.half)
    return fuse(row.large,first,config['fusion'],confidence=confidence,third=second)

def compare(coefficient):
    results={};oracles={}
    for dataset in ['dev','jember']:
        frames={fold:frame(fold,coefficient,dataset) for fold in ['fold_A','fold_B']}
        for name,config in CONFIGS.items():
            row={}
            for fold,data in frames.items():
                predictions=data['large' if name=='Large standalone' else 'old'].tolist() if config is None else [predict(x,config) for x in data.itertuples()]
                if config is not None:assert predictions==[predict(x,config) for x in reversed(list(data.itertuples()))][::-1]
                row[fold]=score(data.text.tolist(),predictions)
                edit_anchor=data.turbo if name=='Old conservative' else data.large
                edits=[edit_stats(a,b) for a,b in zip(edit_anchor,predictions)]
                row[fold]['edits']={k:sum(e[k] for e in edits) for k in edits[0]}
                row[fold]['changed_clips']=sum(a!=b for a,b in zip(edit_anchor,predictions))
                row[fold]['average_corrections_per_clip']=row[fold]['edits']['word_edits']/len(data)
                path=OUT/f'fusion_predictions/{coefficient}/{dataset}/{name}_{fold}.csv';path.parent.mkdir(parents=True,exist_ok=True)
                pd.DataFrame({'clip_id':data.clip_id,'reference':data.text,'transcript':predictions}).to_csv(path,index=False)
            row['aggregate']=aggregate([row[f] for f in frames]);row['worst_fold']=max(row[f]['wer'] for f in frames)
            row['aggregate']['edits']={k:sum(row[f]['edits'][k] for f in frames) for k in row['fold_A']['edits']}
            results.setdefault(name,{'config':config})[dataset]=row
        for fold,data in frames.items():
            oracles.setdefault(dataset,{})[fold]={pair:oracle(data.text.tolist(),*(data[s].tolist() for s in systems)) for pair,systems in [('Large+Turbo',['large','turbo']),('Large+MER',['large','meralion']),('triple',['large','turbo','meralion']),('Large+TurboHalfFull',['large','turbo','half'])]}
        oracles[dataset]['aggregate']={p:aggregate([oracles[dataset][f][p] for f in frames]) for p in oracles[dataset]['fold_A']}
    selection=OUT/'anchor_selection.json';mode=json.loads(selection.read_text()).get('prediction_mode','adapted') if selection.exists() else 'adapted'
    output={'coefficient':coefficient,'prediction_mode':mode,'candidates':results,'oracles':oracles,'oracle_note':'Reference-selected clip hypotheses are diagnostic lower bounds only.'}
    (OUT/f'fusion_comparison_{coefficient}_{mode}.json').write_text(json.dumps(output,indent=2)+'\n')
    for name,r in results.items():print(name,'A/B/aggregate/Jember',*[round(r['dev'][k]['wer'],6) for k in ['fold_A','fold_B','aggregate']],round(r['jember']['aggregate']['wer'],6),'edits',r['dev']['aggregate']['edits'],flush=True)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--coefficient',type=float,default=1);a=p.parse_args();compare(a.coefficient)
