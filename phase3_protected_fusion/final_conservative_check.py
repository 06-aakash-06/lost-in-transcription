"""Three bounded last-submission variants; cached text, teacher-forced confidence."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import sys,json,time,gc,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import torch
from phase3_protected_fusion.data import load
from phase3_protected_fusion.runtime.fusion import fuse,alignment,blocks,collapse_single_runs
from phase3_protected_fusion.runtime.confidence import score_pair
from phase3_protected_fusion.runtime.audio import load_audio
from phase1_whisper_anchor.inference.whisper import WhisperAnchor
from phase1_whisper_anchor.training.metrics import corpus
from phase3_protected_fusion.runtime.meralion import Meralion
OUT=ROOT/'phase3_protected_fusion/results/conservative_final';OUT.mkdir(exist_ok=True)
BASE={'mode':'support','max_span':3,'context':1,'collapse_anchor':True,'operations':['substitution','insertion','deletion']}
THRESHOLDS=[.45,.65,.80]

def release():
    gc.collect()
    if torch.backends.mps.is_available():torch.mps.empty_cache()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)

def jember_frame(fold):
    frame=pd.read_csv(ROOT/'phase1_whisper_anchor/data/jember_holdout.tsv',sep='\t',keep_default_na=False).set_index('clip_id')
    for key,checkpoint in [('full','epoch_1_full'),('anchor','epoch_1_half')]:
        saved=pd.read_csv(ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/{checkpoint}/jember.csv',keep_default_na=False).set_index('clip_id')
        assert saved.index.is_unique and set(saved.index)==set(frame.index)
        saved=saved.loc[frame.index];assert saved.reference.tolist()==frame.text.tolist()
        frame[key]=saved.transcript
    predictions=json.loads((OUT/'jember_meralion_predictions.json').read_text())
    assert set(predictions)==set(frame.index)
    frame['meralion']=[predictions[c]['transcript'] for c in frame.index]
    assert all(predictions[c]['reference']==frame.loc[c,'text'] for c in frame.index)
    return frame.reset_index()

def confidence(dataset):
    started=time.monotonic()
    for fold in ['fold_A','fold_B']:
        frame=load(fold) if dataset=='dev' else jember_frame(fold)
        path=OUT/f'confidence_full_{dataset}_{fold}.json';cache=json.loads(path.read_text()) if path.exists() else {}
        needed=[]
        for row in frame.itertuples():
            if row.clip_id in cache:
                assert cache[row.clip_id]['transcript']==row.full
                continue
            # Scores are only necessary where exact agreement proposes an edit.
            if row.duration_s>30 or fuse(row.full,row.meralion,BASE,third=row.anchor)==row.full:
                cache[row.clip_id]={'transcript':row.full,'anchor':[None]*len(row.full.split())}
            else:needed.append(row)
        if needed:
            engine=WhisperAnchor.from_adapter(str(ROOT/'submission_src/models/whisper_large_v3_turbo'),ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/epoch_1_full','indonesian',1)
            for i,row in enumerate(needed,1):
                scores=score_pair(engine,load_audio(ROOT/row.path),[row.full])[0]
                assert len(scores)==len(row.full.split())
                cache[row.clip_id]={'transcript':row.full,'anchor':scores}
                if i%20==0:
                    write(path,cache);print('confidence',dataset,fold,i,len(needed),'elapsed',round(time.monotonic()-started,1),flush=True)
            del engine;release()
        assert set(cache)==set(frame.clip_id);write(path,cache)
        print('confidence COMPLETE',dataset,fold,round(time.monotonic()-started,1),flush=True)

def jember_meralion():
    frame=pd.read_csv(ROOT/'phase1_whisper_anchor/data/jember_holdout.tsv',sep='\t',keep_default_na=False)
    path=OUT/'jember_meralion_predictions.json';cache=json.loads(path.read_text()) if path.exists() else {}
    remaining=frame[~frame.clip_id.isin(cache)]
    if len(remaining):
        device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        if device=='mps':torch.mps.set_per_process_memory_fraction(.9)
        engine=Meralion(ROOT/'phase2_meralion/models/final_merged',device);started=time.monotonic()
        timings=[]
        for i,row in enumerate(remaining.itertuples(),1):
            before=time.monotonic();pred=engine.transcribe_arrays([load_audio(ROOT/row.path)])[0];timings.append(time.monotonic()-before)
            cache[row.clip_id]={'reference':row.text,'transcript':pred};write(path,cache)
            if i%10==0:
                elapsed=time.monotonic()-started;eta=sum(timings)/len(timings)*(len(remaining)-i)
                print('Jember MERaLiON',i,len(remaining),'elapsed',round(elapsed,1),'remaining estimate',round(eta,1),flush=True)
                if elapsed+eta>1200:
                    write(OUT/'jember_skipped.json',{'reason':'Measured decoding projects >20 min, leaving insufficient bounded-task margin.','completed':len(cache),'total':len(frame)})
                    print('Jember SKIPPED: projected >20 min',flush=True);return
        del engine;release()
    assert len(cache)==162
    print('Jember MERaLiON COMPLETE',flush=True)

def edit_stats(anchor,pred):
    s=i=d=0
    for a,b,equal in alignment(anchor.split(),pred.split()):
        if equal:continue
        if a is None:i+=1
        elif b is None:d+=1
        else:s+=1
    return {'edit_spans':len(blocks(anchor.split(),pred.split())),'word_edits':s+i+d,'substitutions':s,'insertions':i,'deletions':d}

def score(dataset):
    variants={'R3-full':None,'R3-full-safe':None,'R3-half':None,'MERaLiON':None,'A':BASE,'B':{**BASE,'operations':['substitution'],'strict_substitutions':True}}
    thresholds=THRESHOLDS
    if dataset=='jember':
        # Freeze C using dev only. Jember never selects a confidence threshold.
        dev=json.loads((OUT/'dev_metrics.json').read_text())
        thresholds=[min(THRESHOLDS,key=lambda t:dev[f'C_{t}']['aggregate']['wer'])]
    variants.update({f'C_{t}':{**BASE,'mode':'support_confidence','threshold':t,'allow_deletion':True} for t in thresholds})
    records={};allrefs=[];allpred={name:[] for name in variants};fold_edits={}
    for fold in ['fold_A','fold_B']:
        frame=load(fold) if dataset=='dev' else jember_frame(fold)
        cache=json.loads((OUT/f'confidence_full_{dataset}_{fold}.json').read_text())
        allrefs+=frame.text.tolist()
        for name,cfg in variants.items():
            if cfg is None:
                pred=frame[{'R3-full':'full','R3-full-safe':'full','R3-half':'anchor','MERaLiON':'meralion'}[name]].tolist()
                if name=='R3-full-safe':pred=[collapse_single_runs(p) for p in pred]
            else:
                pred=[fuse(x.full,x.meralion,cfg,cache[x.clip_id],x.anchor) for x in frame.itertuples()]
                repeat=[fuse(x.full,x.meralion,cfg,cache[x.clip_id],x.anchor) for x in reversed(list(frame.itertuples()))][::-1]
                assert pred==repeat
            stats=corpus(frame.text.tolist(),pred)
            if cfg is not None:
                # Common catastrophic-loop safety is held fixed, separate from
                # model-supported corrections, so B has strictly no indels.
                edits=[edit_stats(collapse_single_runs(a),p) for a,p in zip(frame.full,pred)]
                stats['edits']={k:sum(x[k] for x in edits) for k in edits[0]}
                safety=[edit_stats(a,collapse_single_runs(a)) for a in frame.full]
                stats['common_repeat_safety']={k:sum(x[k] for x in safety) for k in safety[0]}
                if name=='B':assert stats['edits']['insertions']==stats['edits']['deletions']==0
            records.setdefault(name,{})[fold]=stats;allpred[name]+=pred
            pd.DataFrame({'clip_id':frame.clip_id,'reference':frame.text,'transcript':pred}).to_csv(OUT/f'{dataset}_{name}_{fold}.csv',index=False)
    for name in variants:
        records[name]['aggregate']=corpus(allrefs,allpred[name])
        if variants[name] is not None:
            records[name]['aggregate']['edits']={k:sum(records[name][f]['edits'][k] for f in ['fold_A','fold_B']) for k in records[name]['fold_A']['edits']}
            records[name]['aggregate']['common_repeat_safety']={k:sum(records[name][f]['common_repeat_safety'][k] for f in ['fold_A','fold_B']) for k in records[name]['fold_A']['common_repeat_safety']}
    write(OUT/f'{dataset}_metrics.json',records)
    print(dataset,json.dumps(records,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['confidence_dev','jember_meralion','confidence_jember','score_dev','score_jember']);a=p.parse_args()
    if a.step.startswith('confidence_'):confidence(a.step.split('_')[1])
    elif a.step=='jember_meralion':jember_meralion()
    elif a.step.startswith('score_'):score(a.step.split('_')[1])
