"""Small predeclared interpretable fusion search; official corpus WER."""
import sys,json,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from phase3_protected_fusion.data import load
from phase3_protected_fusion.runtime.fusion import fuse
from phase1_whisper_anchor.training.metrics import counts
HERE=Path(__file__).resolve().parent

def main():
    data={f:load(f) for f in ['fold_A','fold_B']}
    confidence={f:json.loads((HERE/f'results/confidence_{f}.json').read_text()) for f in data}
    configs=[]
    for threshold,span,edits in itertools.product([.45,.65],[3,5],[False,True]):
        configs.append(dict(name=f'bounded_t{threshold}_s{span}_edits{edits}',mode='confidence',threshold=threshold,margin=None,max_span=span,context=1,collapse_anchor=True,operations=['substitution','insertion','deletion'] if edits else ['substitution'],allow_deletion=edits))
    cache={};rows=[]
    for config in configs:
        row={'configuration':config['name'],'config':json.dumps(config,sort_keys=True)};total=[0]*4
        for fold,frame in data.items():
            sums=[0]*4;predictions=[]
            for x in frame.itertuples():
                c=confidence[fold].get(x.clip_id)
                pred=fuse(x.anchor,x.meralion,config,c,getattr(x,config['third']) if config.get('third') else None)
                ck=(x.text,pred)
                if ck not in cache:cache[ck]=counts(*ck)
                for k,v in enumerate(cache[ck]):sums[k]+=v
                predictions.append(pred)
            row[fold]=(sum(sums[1:])/sums[0]);row[fold+'_counts']=json.dumps(sums)
            total=[a+b for a,b in zip(total,sums)]
            if config['name']!='anchor':
                pd.DataFrame({'clip_id':frame.clip_id,'reference':frame.text,'transcript':predictions}).to_csv(HERE/f"results/{config['name']}_{fold}.csv",index=False)
        row['aggregate']=sum(total[1:])/total[0];row['worst']=max(row['fold_A'],row['fold_B']);rows.append(row)
        print(config['name'],row['fold_A'],row['fold_B'],row['aggregate'],flush=True)
    out=pd.DataFrame(rows).sort_values(['aggregate','worst']);out.to_csv(HERE/'results/fusion_extended.csv',index=False)
    print(out[['configuration','fold_A','fold_B','aggregate']].head(10).to_string(index=False))
if __name__=='__main__':main()
