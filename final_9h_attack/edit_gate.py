"""One cross-conversation logistic edit gate; no sklearn in deployment."""
import sys,json,argparse,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from final_9h_attack.compare_fusion import frame
from final_9h_attack.runtime.consensus import consensus,proposals,gate_features,FEATURES
from overnight_attack.evidence import score,aggregate,error
from phase3_protected_fusion.final_conservative_check import edit_stats
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def confidence(coefficient,dataset,fold):return json.loads((OUT/f'confidence/{coefficient}/{dataset}_{fold}.json').read_text())

def observations(data,cache):
    features=[];labels=[]
    for row in data.itertuples():
        if row.large_raw!=row.large:continue
        entry=cache[row.clip_id];assert entry['transcript']==row.large
        baseline=sum(error(row.text,row.large)[1:])
        for i,candidate in proposals(row.large,row.turbo,row.meralion,0):
            words=row.large.split();words[i]=candidate
            changed=sum(error(row.text,' '.join(words))[1:])
            features.append(gate_features(row.large,row.turbo,row.meralion,i,entry['anchor']))
            labels.append(int(changed<baseline))
    return np.asarray(features,dtype=float),np.asarray(labels)

def fit(x,y):
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-6]=1.
    model=LogisticRegression(C=1.,class_weight='balanced',solver='lbfgs',max_iter=1000,random_state=1337).fit((x-mean)/scale,y)
    gate={'features':FEATURES,'mean':mean.tolist(),'scale':scale.tolist(),'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),'threshold':.5,'training_edits':len(y),'beneficial_edits':int(y.sum())}
    raw=gate['intercept']+((x-mean)/scale)@np.asarray(gate['coef'])
    assert np.allclose(raw,model.decision_function((x-mean)/scale),atol=1e-10)
    return gate

def evaluate(coefficient):
    data={fold:frame(fold,coefficient) for fold in ['fold_A','fold_B']}
    cache={fold:confidence(coefficient,'dev',fold) for fold in data}
    observations_by_fold={f:observations(data[f],cache[f]) for f in data}
    models={test:fit(*observations_by_fold[train]) for train,test in [('fold_A','fold_B'),('fold_B','fold_A')]}
    configs={'C consensus':None,**{f'D low confidence {t}':t for t in [.6,.8,.95]},'Cross-fold logistic': 'gate'}
    results={}
    for name,choice in configs.items():
        metrics={}
        for fold,table in data.items():
            gate=models[fold] if choice=='gate' else None;threshold=choice if isinstance(choice,float) else None
            pred=[consensus(r.large,r.turbo,r.meralion,confidence=cache[fold][r.clip_id],threshold=threshold,gate=gate) for r in table.itertuples()]
            metrics[fold]=score(table.text.tolist(),pred)
            edits=[edit_stats(r.large,p) for r,p in zip(table.itertuples(),pred)]
            metrics[fold]['edits']={k:sum(e[k] for e in edits) for k in edits[0]}
            path=OUT/f'gate_predictions/{name}_{fold}.csv';path.parent.mkdir(parents=True,exist_ok=True)
            pd.DataFrame({'clip_id':table.clip_id,'reference':table.text,'transcript':pred}).to_csv(path,index=False)
        metrics['aggregate']=aggregate([metrics[f] for f in data]);results[name]=metrics
    baseline=results['C consensus'];gated=results['Cross-fold logistic']
    passes=all(gated[f]['wer']<baseline[f]['wer'] for f in data) and gated['aggregate']['wer']<=baseline['aggregate']['wer']-.001
    x=np.concatenate([o[0] for o in observations_by_fold.values()]);y=np.concatenate([o[1] for o in observations_by_fold.values()])
    final=fit(x,y)
    output={'results':results,'cross_fold_gates':models,'final_gate':final,'passes_two_direction_gate':passes,
        'policy':'Keep only if both opposite-fold tests improve and aggregate gain >= .001; Jember cannot fit the gate or threshold.'}
    # Frozen gates/thresholds are diagnosed on Jember, with no holdout labels fitted.
    for name,choice in configs.items():
        held={}
        for fold in data:
            table=frame(fold,coefficient,'jember');hc=confidence(coefficient,'jember',fold)
            # Cross-fold fitted model for each fold checkpoint, not an OOF-fitted diagnostic.
            gate=models[fold] if choice=='gate' else None;threshold=choice if isinstance(choice,float) else None
            pred=[consensus(r.large,r.turbo,r.meralion,confidence=hc[r.clip_id],threshold=threshold,gate=gate) for r in table.itertuples()]
            held[fold]=score(table.text.tolist(),pred)
        results[name]['jember']=aggregate(list(held.values()))
    (OUT/'edit_gate_comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--coefficient',type=float,required=True);a=p.parse_args();evaluate(a.coefficient)
