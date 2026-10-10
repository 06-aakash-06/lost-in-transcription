"""Bounded cached-OOF analysis for the last competition submission."""
import sys,json,time
from pathlib import Path
from functools import lru_cache
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase3_protected_fusion.data import load
from phase3_protected_fusion.runtime.fusion import fuse,collapse_single_runs
from phase3_protected_fusion.runtime.edit_gate import FEATURES,proposals,apply_edits,gated_fuse
from phase1_whisper_anchor.training.official_score import normalize_text
from phase1_whisper_anchor.training.metrics import counts
OUT=ROOT/'phase3_protected_fusion/results/final_quick';OUT.mkdir(exist_ok=True)

@lru_cache(None)
def error(ref,pred):
    a,b=normalize_text(ref).split(),normalize_text(pred).split();row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]

data={f:load(f) for f in ['fold_A','fold_B']}
confidence={f:json.loads((ROOT/f'phase3_protected_fusion/results/confidence_{f}.json').read_text()) for f in data}
base=json.loads((ROOT/'phase3_protected_fusion/configs/candidate_B.json').read_text())['fusion']
rows=[];predictions={}
def evaluate(name,outputs,details):
    row={'name':name,**details};total=nwords=0
    for fold,frame in data.items():
        errors=sum(error(r,p) for r,p in zip(frame.text,outputs[fold]));words=sum(len(normalize_text(r).split()) for r in frame.text)
        row[fold]=errors/words;total+=errors;nwords+=words
        pd.DataFrame({'clip_id':frame.clip_id,'reference':frame.text,'transcript':outputs[fold]}).to_csv(OUT/f'{name}_{fold}.csv',index=False)
    row['aggregate']=total/nwords;rows.append(row);predictions[name]=outputs
    print(json.dumps(row),flush=True)

for anchor,third in [('anchor','full'),('full','anchor'),('ratio4','anchor'),('ratio4','full'),('auto','anchor'),('auto','full'),('anchor','ratio4')]:
    # Saved acoustic scores belong to the R3-half hypothesis only. Never reuse
    # them for a different anchor; those permutations use exact voter support.
    cfg=base if anchor=='anchor' else {**base,'mode':'support'}
    outputs={f:[fuse(getattr(x,anchor),x.meralion,cfg,confidence[f].get(x.clip_id) if anchor=='anchor' else None,getattr(x,third)) for x in frame.itertuples()] for f,frame in data.items()}
    evaluate(f'permutation_{anchor}_{third}',outputs,{'anchor':anchor,'third':third,'mode':cfg['mode'],'confidence_available':anchor=='anchor'})

variants={
 'substitutions':{'operations':['substitution']},
 'safe_insertions':{'operations':['substitution','insertion']},
 'short_span':{'max_span':2},
 'support_required':{'mode':'support'},
 'lower_confidence':{'threshold':.45},
 'restricted_deletions':{'operations':['substitution','insertion'],'duplicate_deletions':True},
 'repetition_rescue':{'repeat_swap':True},
}
for name,update in variants.items():
    cfg={**base,**update};outputs={}
    for f,frame in data.items():
        out=[]
        for x in frame.itertuples():
            pred=fuse(x.anchor,x.meralion,cfg,confidence[f].get(x.clip_id),x.full)
            if update.get('duplicate_deletions'):
                # Only duplicate removals that both other models support.
                from phase3_protected_fusion.runtime.fusion import blocks,key
                a=pred.split();b=x.meralion.split();supported=fuse(pred,x.meralion,{**base,'mode':'support','operations':['deletion'],'max_span':1},third=x.full)
                deletes=[]
                for q in blocks(a,supported.split()):
                    if q.end-q.start==1 and q.other_start==q.other_end and ((q.start and key(a[q.start])==key(a[q.start-1])) or (q.end<len(a) and key(a[q.start])==key(a[q.end]))):deletes.append((q.start,q.end,[]))
                pred=apply_edits(pred,deletes)
            out.append(pred)
        outputs[f]=out
    evaluate('rule_'+name,outputs,{'config':cfg})

training={}
for f,frame in data.items():
    features=[];labels=[]
    for x in frame.itertuples():
        e0=error(x.text,x.anchor)
        for start,end,candidate,feat,baseline in proposals(x.anchor,x.meralion,confidence[f].get(x.clip_id),x.full):
            e1=error(x.text,apply_edits(x.anchor,[(start,end,candidate)]))
            features.append(feat);labels.append(int(e1<e0))
    training[f]=(np.asarray(features),np.asarray(labels))
    print('gate_training',f,len(labels),sum(labels),flush=True)

models={}
for train_fold,test_fold in [('fold_A','fold_B'),('fold_B','fold_A')]:
    features,labels=training[train_fold];scale=StandardScaler().fit(features)
    model=LogisticRegression(C=1.,max_iter=1000,random_state=1337).fit(scale.transform(features),labels)
    models[test_fold]={'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),'mean':scale.mean_.tolist(),'scale':scale.scale_.tolist(),'features':FEATURES,'training_fold':train_fold}
for scope in ['baseline','all']:
    outputs={f:[gated_fuse(x.anchor,x.meralion,confidence[f].get(x.clip_id),x.full,{**models[f],'scope':scope}) for x in frame.itertuples()] for f,frame in data.items()}
    evaluate('gate_'+scope,outputs,{'crossfit':True,'features':FEATURES})

baseline=rows[0]
accepted=[r for r in rows if r['name'].startswith(('rule_','gate_')) and r['fold_A']<baseline['fold_A'] and r['fold_B']<baseline['fold_B'] and baseline['aggregate']-r['aggregate']>=.0015]
selected=min(accepted,key=lambda r:r['aggregate']) if accepted else baseline
if selected['name'].startswith('gate_'):
    features=np.concatenate([training[f][0] for f in data]);labels=np.concatenate([training[f][1] for f in data]);scale=StandardScaler().fit(features)
    model=LogisticRegression(C=1.,max_iter=1000,random_state=1337).fit(scale.transform(features),labels)
    gate={'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),'mean':scale.mean_.tolist(),'scale':scale.scale_.tolist(),'features':FEATURES,'scope':selected['name'][5:]}
    (OUT/'final_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    (OUT/'crossfit_gate_weights.json').write_text(json.dumps(models,indent=2)+'\n')
# Verify integer scorer against official S/D/I for selected outputs only.
for f,frame in data.items():
    for r,p in zip(frame.text,predictions[selected['name']][f]):assert error(r,p)==sum(counts(r,p)[1:])
oracle={};total=words=0
for f,frame in data.items():
    e=sum(min(error(x.text,x.anchor),error(x.text,x.meralion),error(x.text,x.full)) for x in frame.itertuples());n=sum(len(normalize_text(r).split()) for r in frame.text);oracle[f]=e/n;total+=e;words+=n
oracle['aggregate']=total/words
result={'selected':selected,'baseline':baseline,'oracle':oracle,'trials':rows,'gate_policy':'Reject unless both held-out directions improve and aggregate gain >= .0015','alternative_anchor_note':'No saved confidence for alternate anchors; exact-support fusion evaluated without mismatched scores.'}
(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print('SELECTED',json.dumps(selected),flush=True)
