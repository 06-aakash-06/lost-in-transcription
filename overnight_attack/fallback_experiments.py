"""Small compatible delta-soup set, reference-free adaptive decode, and n-gram rescoring."""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from overnight_attack.soup import average
from overnight_attack.evidence import score,aggregate,oracle
from overnight_attack.evaluate import Engine
from phase4_stronger_anchor.audit import aligned
from phase3_protected_fusion.runtime.fusion import collapse_single_runs,fuse,repetition
from phase3_protected_fusion.runtime.confidence import score_pair
from phase1_whisper_anchor.inference.audio import load_audio
from phase1_whisper_anchor.training.official_score import normalize_text
from phase1_whisper_anchor.training.text import to_reference_style
HERE=ROOT/'overnight_attack';OUT=HERE/'results';FOLDS=['fold_A','fold_B']

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def command(name,args,deadline,expected=None):
    if expected and expected.exists():return True
    for attempt in range(3):
        if time.time()>deadline-30:return False
        with (OUT/f'{name}_retry{attempt+1}.log').open('a') as log:
            process=subprocess.Popen([sys.executable,*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:code=process.wait(timeout=max(30,deadline-time.time()))
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=60);return False
        if code==0 and (expected is None or expected.exists()):return True
        print('RETRY',name,attempt+1,'exit',code,flush=True)
    return False

def data(fold,held=False):
    name='jember_holdout' if held else f'{fold}_valid'
    return pd.read_csv(ROOT/f'phase1_whisper_anchor/data/{name}.tsv',sep='\t',keep_default_na=False)

def cached(fold,checkpoint,frame,held=False,ratio=3):
    name='jember' if held else 'valid'
    return list(map(collapse_single_runs,aligned(ROOT/f'phase1_whisper_anchor/runs/indonesian_x{ratio}_{fold}/{checkpoint}/{name}.csv',frame)))

def evaluate_adapter(name,adapter,base,dtype,fold,held,deadline):
    manifest_name='jember_holdout' if held else f'{fold}_valid'
    output=OUT/'soup_predictions'/name/f'{fold}_{manifest_name}.json'
    okay=command('soup_eval_'+name+'_'+manifest_name,[HERE/'evaluate.py','--base',base,'--checkpoint',adapter,'--dtype',dtype,'--manifest',ROOT/f'phase1_whisper_anchor/data/{manifest_name}.tsv','--output',output,'--batch-size','4'],deadline,output)
    if not okay:return None
    return list(map(collapse_single_runs,aligned(output.with_suffix('.csv'),data(fold,held))))

def soup(deadline,large=False):
    trials=[];baseline={};heldbase={}
    if large:
        pair_sets=[('large_half_full','large','epoch_1_half','large','epoch_1_full')]
        dtype='bfloat16';base=HERE/'models/large_v3_base';weights=[.5]
        selection=json.loads((OUT/'large_v3_two_fold.json').read_text())['selected_checkpoint']
    else:
        pair_sets=[('R3_half_full',3,'epoch_1_half',3,'epoch_1_full'),('R3_full_R4_full',3,'epoch_1_full',4,'epoch_1_full')]
        dtype='float32';base=ROOT/'submission_src/models/whisper_large_v3_turbo';weights=[.25,.5,.75]
    for fold in FOLDS:
        frame=data(fold);held=data(fold,True)
        if large:
            folder=OUT/'predictions'/f'large_v3_{fold}'/selection
            baseline[fold]=score(frame.text.tolist(),list(map(collapse_single_runs,aligned(folder/f'{fold}_valid.csv',frame))))
            heldbase[fold]=score(held.text.tolist(),list(map(collapse_single_runs,aligned(folder/'jember_holdout.csv',held))))
        else:
            # The strongest existing aggregate Turbo parent is Ratio-4 full.
            baseline[fold]=score(frame.text.tolist(),cached(fold,'epoch_1_full',frame,ratio=4))
            heldbase[fold]=score(held.text.tolist(),cached(fold,'epoch_1_full',held,True,4))
    for family,left_ratio,left_cp,right_ratio,right_cp in pair_sets:
        for weight in weights:
            if time.time()>deadline-180:break
            trial={'family':family,'left_weight':weight,'metrics':{},'jember':{},'adapters':{}}
            name=f'{family}_w{weight}'
            for fold in FOLDS:
                if large:
                    left=HERE/f'runs/large_v3_{fold}'/left_cp;right=HERE/f'runs/large_v3_{fold}'/right_cp
                else:
                    left=ROOT/f'phase1_whisper_anchor/runs/indonesian_x{left_ratio}_{fold}'/left_cp
                    right=ROOT/f'phase1_whisper_anchor/runs/indonesian_x{right_ratio}_{fold}'/right_cp
                adapter=HERE/'runs/soup_adapters'/name/fold
                try:average(left,right,adapter,weight)
                except Exception as exc:trial['failed']=repr(exc);break
                prediction=evaluate_adapter(name,adapter,base,dtype,fold,False,deadline)
                if prediction is None:trial['failed']='evaluation failed after retries/time budget';break
                trial['metrics'][fold]=score(data(fold).text.tolist(),prediction);trial['adapters'][fold]=str(adapter)
            if len(trial['metrics'])==2:
                trial['metrics']['aggregate']=aggregate([trial['metrics'][f] for f in FOLDS])
                parent=aggregate([baseline[f] for f in FOLDS])
                trial['passes_two_fold_guard']=all(trial['metrics'][f]['wer']<=baseline[f]['wer']+.001 for f in FOLDS) and trial['metrics']['aggregate']['wer']<=parent['wer']-.0005
                if trial['passes_two_fold_guard']:
                    for fold in FOLDS:
                        held_prediction=evaluate_adapter(name,Path(trial['adapters'][fold]),base,dtype,fold,True,deadline)
                        if held_prediction is None:trial['passes_two_fold_guard']=False;break
                        trial['jember'][fold]=score(data(fold,True).text.tolist(),held_prediction)
                    if len(trial['jember'])==2:
                        trial['jember']['aggregate']=aggregate([trial['jember'][f] for f in FOLDS])
                        trial['passes_two_fold_guard'] &= trial['jember']['aggregate']['wer']<=aggregate([heldbase[f] for f in FOLDS])['wer']+.015
            trials.append(trial)
            write(OUT/('large_soup_trials.json' if large else 'soup_trials.json'),trials)
            print('SOUP',json.dumps(trial),flush=True)
    winners=[r for r in trials if r.get('passes_two_fold_guard')]
    completed=[r for r in trials if 'aggregate' in r['metrics']]
    best=min(winners or completed,key=lambda r:r['metrics']['aggregate']['wer']) if completed else None
    final_soup=None
    if winners and time.time()<deadline-180:
        winner=min(winners,key=lambda r:r['metrics']['aggregate']['wer'])
        family=next(row for row in pair_sets if row[0]==winner['family']);_,lr,lc,rr,rc=family
        if large:left=HERE/'runs/final_large_v3'/lc;right=HERE/'runs/final_large_v3'/rc
        else:
            left=ROOT/'phase1_whisper_anchor/runs/final_indonesian_x3'/lc
            right=(ROOT/'phase1_whisper_anchor/runs/final_indonesian_x3' if rr==3 else HERE/'runs/final_turbo_ratio4')/rc
        if (left/'adapter_model.safetensors').exists() and (right/'adapter_model.safetensors').exists():
            final_soup=HERE/'runs'/('final_large_soup' if large else 'final_turbo_soup')
            average(left,right,final_soup,winner['left_weight'])
            merged=HERE/'models'/('final_large_soup' if large else 'final_turbo_soup')
            command('merge_'+merged.name,[HERE/'merge.py','--base-model',base,'--adapter',final_soup,'--output',merged],deadline,merged/'merge_validation.json')
    report='No soup improved both-fold robustness enough to keep.'
    if best:
        m=best['metrics'];report=f"Best tested {best['family']} left weight {best['left_weight']}: A {m['fold_A']['wer']:.6f}, B {m['fold_B']['wer']:.6f}, aggregate {m['aggregate']['wer']:.6f}; keep={bool(winners)}."
    write(OUT/'soup_summary.json',{'report':report,'best':best,'kept':bool(winners),'final_adapter':str(final_soup) if final_soup else None,'baseline':baseline,'trials':trials,
        'final_R4_schedule_note':'A cross-ratio final soup uses the available two-epoch-scheduled final R4 first-epoch checkpoint; OOF R4 parents had a one-epoch schedule. Final Jember validation is required before deployment.' if not large else None})

def release():
    gc.collect()
    if torch.backends.mps.is_available():torch.mps.empty_cache()

def adaptive(deadline):
    pools={};frames={};confidence={};alternatives={};trials=[]
    for fold in FOLDS:
        frame=data(fold);frames[fold]=frame
        raw=aligned(ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/epoch_1_full/valid.csv',frame)
        meralion=aligned(ROOT/f'phase2_meralion/results/pred_epoch_1_half_{fold}.csv',frame)
        conservative=aligned(ROOT/f'phase3_protected_fusion/results/conservative_final/dev_B_{fold}.csv',frame)
        pools[fold]={'raw':raw,'safe':list(map(collapse_single_runs,raw)),'meralion':meralion,'conservative':conservative}
        path=OUT/f'adaptive_confidence_{fold}.json'
        saved=json.loads(path.read_text()) if path.exists() else {}
        previous=json.loads((ROOT/f'phase3_protected_fusion/results/conservative_final/confidence_full_dev_{fold}.json').read_text())
        for clip,text in zip(frame.clip_id,raw):
            if clip in previous and previous[clip]['transcript']==text and any(x is not None for x in previous[clip]['anchor']):saved.setdefault(clip,previous[clip])
        needed=[(row,text) for row,text in zip(frame.itertuples(),raw) if row.clip_id not in saved]
        if needed:
            engine=Engine(ROOT/'submission_src/models/whisper_large_v3_turbo',ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/epoch_1_full','float32','indonesian',1,1)
            for row,text in needed:
                if time.time()>deadline-120:break
                scores=score_pair(engine,load_audio(ROOT/row.path),[text])[0]
                saved[row.clip_id]={'transcript':text,'anchor':scores};write(path,saved)
            del engine;release()
        means={}
        for row,text in zip(frame.itertuples(),raw):
            values=[x for x in saved.get(row.clip_id,{}).get('anchor',[]) if x is not None]
            means[row.clip_id]=sum(values)/len(values) if values else -float('inf') if repetition(text)>.2 else None
        confidence[fold]=means
        eligible=sorted([c for c in frame.clip_id if means[c] is not None],key=lambda c:means[c])
        # At most the lowest-confidence 35%; clip selection uses predictions,
        # acoustic confidence and repetition only, never reference labels.
        selected=set(eligible[:max(1,math.ceil(.35*len(eligible)))])
        selected.update(c for c,t in zip(frame.clip_id,raw) if repetition(t)>.2)
        sub=frame[frame.clip_id.isin(selected)];manifest_path=HERE/'cache'/f'adaptive_{fold}.tsv';sub.to_csv(manifest_path,sep='\t',index=False)
        for name,beams,language in [('beam3',3,'indonesian'),('beam5',5,'indonesian'),('auto',1,'auto')]:
            output=OUT/'adaptive_predictions'/f'{fold}_{name}.json'
            okay=command('adaptive_'+fold+'_'+name,[HERE/'evaluate.py','--base',ROOT/'submission_src/models/whisper_large_v3_turbo',
                '--checkpoint',ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/epoch_1_full','--manifest',manifest_path,
                '--output',output,'--beams',str(beams),'--language',language,'--batch-size','2'],deadline,output)
            if okay:
                predictions=pd.read_csv(output.with_suffix('.csv'),keep_default_na=False).set_index('clip_id')
                assert set(predictions.index)==set(selected)
                alternatives.setdefault(name,{})[fold]={c:predictions.loc[c,'transcript'] for c in selected}
    for name,byfold in alternatives.items():
        if len(byfold)!=2:continue
        for quantile in [.20,.35]:
            metrics={};bounds={};pure={};changed={}
            for fold in FOLDS:
                frame=frames[fold];pool=pools[fold];means=confidence[fold]
                ranked=sorted([c for c in frame.clip_id if means[c] is not None],key=lambda c:means[c]);approved=set(ranked[:max(1,math.ceil(quantile*len(ranked)))])
                approved.update(c for c,t in zip(frame.clip_id,pool['raw']) if repetition(t)>.2)
                predictions=[];alt_full=[]
                cfg={'mode':'support','max_span':3,'context':1,'collapse_anchor':True,'operations':['substitution'],'strict_substitutions':True}
                for clip,primary,corrective,old in zip(frame.clip_id,pool['raw'],pool['meralion'],pool['conservative']):
                    alt=byfold[fold].get(clip,primary);alt_full.append(collapse_single_runs(alt))
                    predictions.append(fuse(old,corrective,cfg,third=alt) if clip in approved and clip in byfold[fold] else old)
                metrics[fold]=score(frame.text.tolist(),predictions);bounds[fold]=oracle(frame.text.tolist(),pool['safe'],alt_full)
                pure[fold]=score(frame.text.tolist(),alt_full);changed[fold]=sum(a!=b for a,b in zip(predictions,pool['conservative']))
                pd.DataFrame({'clip_id':frame.clip_id,'reference':frame.text,'transcript':predictions}).to_csv(OUT/f'adaptive_{name}_q{quantile}_{fold}.csv',index=False)
            metrics['aggregate']=aggregate([metrics[f] for f in FOLDS]);bounds['aggregate']=aggregate([bounds[f] for f in FOLDS]);pure['aggregate']=aggregate([pure[f] for f in FOLDS])
            trials.append({'alternative':name,'low_confidence_quantile':quantile,'metrics':metrics,'pair_oracle':bounds,'raw_alternative_on_selected_clips':pure,'changed_clips':changed})
    baseline={f:score(frames[f].text.tolist(),pools[f]['conservative']) for f in FOLDS};baseline['aggregate']=aggregate([baseline[f] for f in FOLDS])
    for trial in trials:trial['keep']=all(trial['metrics'][f]['wer']<=baseline[f]['wer'] for f in FOLDS) and trial['metrics']['aggregate']['wer']<baseline['aggregate']['wer']-.0005
    candidates=[t for t in trials if t['keep']]
    best=min(candidates or trials,key=lambda t:t['metrics']['aggregate']['wer']) if trials else None
    report='No completed two-fold adaptive candidate.'
    if best:
        m=best['metrics'];report=f"{best['alternative']} on lowest {best['low_confidence_quantile']:.0%} acoustic confidence with MERaLiON-supported edits: A {m['fold_A']['wer']:.6f}, B {m['fold_B']['wer']:.6f}, aggregate {m['aggregate']['wer']:.6f}; keep={bool(candidates)}."
    write(OUT/'adaptive_summary.json',{'report':report,'best':best,'trials':trials,'baseline':baseline,'keep':bool(candidates),
        'note':'Greedy transcripts reused; only confidence and selected alternative decodes computed. No global switch or reference-based clip selection.'})

def nbest(deadline):
    # Reuse the small generated candidate set before considering another
    # expensive decode. Candidates are Whisper outputs, never rewritten text.
    from collections import defaultdict
    frames={};records={};baselines={};lms={};score_caches={}
    for fold in FOLDS:
        frame=data(fold);frames[fold]=frame
        raw=aligned(ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/epoch_1_full/valid.csv',frame)
        baselines[fold]=list(map(collapse_single_runs,raw));options={c:[t] for c,t in zip(frame.clip_id,raw)}
        for mode in ['beam3','beam5','auto']:
            path=OUT/'adaptive_predictions'/f'{fold}_{mode}.csv'
            if not path.exists():continue
            f=pd.read_csv(path,keep_default_na=False)
            for row in f.itertuples():
                if row.transcript not in options[row.clip_id]:options[row.clip_id].append(row.transcript)
        # Generate actual top-3 beam hypotheses on the already-selected short
        # validation clips. Native long-form remains on the cached baseline.
        nbest_path=OUT/f'whisper_nbest_{fold}.json'
        nbest_cache=json.loads(nbest_path.read_text()) if nbest_path.exists() else {}
        candidates=[row for row in frame.itertuples() if row.duration_s<=30 and len(options[row.clip_id])>1 and row.clip_id not in nbest_cache]
        if candidates and time.time()<deadline-300:
            engine=Engine(ROOT/'submission_src/models/whisper_large_v3_turbo',ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/epoch_1_full','float32','indonesian',5,1)
            try:
                for row in candidates:
                    if time.time()>deadline-300:break
                    wave=load_audio(ROOT/row.path)
                    features=engine.processor.feature_extractor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
                    with torch.inference_mode():
                        ids=engine.model.generate(input_features=features.input_features.to(engine.device,dtype=engine.dtype),attention_mask=features.attention_mask.to(engine.device),
                            language='indonesian',task='transcribe',num_beams=5,num_return_sequences=3,do_sample=False,return_timestamps=False,max_new_tokens=440)
                    texts=engine.processor.batch_decode(ids,skip_special_tokens=True)
                    assert len(texts)==3,'Whisper API did not return three beam hypotheses'
                    nbest_cache[row.clip_id]=texts;write(nbest_path,nbest_cache)
            except Exception as exc:
                write(OUT/f'nbest_generation_failure_{fold}.json',{'error':repr(exc),'completed_clips':len(nbest_cache),'fallback':'Use already cached Whisper candidate set; no prolonged API debugging.'})
            finally:del engine;release()
        for clip,texts in nbest_cache.items():
            for text in texts:
                if text not in options[clip]:options[clip].append(text)
        training=pd.read_csv(ROOT/f'phase1_whisper_anchor/data/{fold}_train.tsv',sep='\t',keep_default_na=False)
        uni=Counter();bi=Counter()
        for row in training.itertuples():
            text=row.text if row.domain=='competition' else to_reference_style(row.text)
            words=['<s>']+normalize_text(text).split()+['</s>'];uni.update(words[:-1]);bi.update(zip(words,words[1:]))
        vocabulary=set(uni)|{'</s>','<unk>'};lms[fold]=(uni,bi,len(vocabulary))
        path=OUT/f'nbest_acoustic_{fold}.json';cache=json.loads(path.read_text()) if path.exists() else {}
        needed=[]
        for row in frame.itertuples():
            if row.duration_s>30 or len(options[row.clip_id])<2:continue
            if row.clip_id not in cache or cache[row.clip_id]['texts']!=options[row.clip_id]:needed.append(row)
        if needed:
            engine=Engine(ROOT/'submission_src/models/whisper_large_v3_turbo',ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/epoch_1_full','float32','indonesian',1,1)
            for row in needed:
                if time.time()>deadline-120:break
                texts=options[row.clip_id];scores=score_pair(engine,load_audio(ROOT/row.path),texts)
                means=[sum(v)/len(v) if v and all(x is not None for x in v) else None for v in scores]
                cache[row.clip_id]={'texts':texts,'mean_acoustic_logp':means};write(path,cache)
            del engine;release()
        records[fold]=options;score_caches[fold]=cache
    trials=[]
    for weight in [.03,.08]:
        metrics={}
        for fold in FOLDS:
            uni,bi,vocab=lms[fold];pred=[]
            for clip,original in zip(frames[fold].clip_id,baselines[fold]):
                cached_scores=score_caches[fold].get(clip)
                if not cached_scores:pred.append(original);continue
                ranked=[]
                for text,acoustic in zip(cached_scores['texts'],cached_scores['mean_acoustic_logp']):
                    if acoustic is None:continue
                    words=['<s>']+normalize_text(text).split()+['</s>'];pairs=list(zip(words,words[1:]))
                    lm=sum(math.log((bi[a,b]+.1)/(uni[a]+.1*vocab)) for a,b in pairs)/max(1,len(pairs))
                    ranked.append((acoustic+weight*lm,text))
                pred.append(collapse_single_runs(max(ranked,key=lambda x:x[0])[1]) if ranked else original)
            metrics[fold]=score(frames[fold].text.tolist(),pred)
        metrics['aggregate']=aggregate([metrics[f] for f in FOLDS]);trials.append({'lm_weight':weight,'metrics':metrics})
    baseline={f:score(frames[f].text.tolist(),baselines[f]) for f in FOLDS};baseline['aggregate']=aggregate([baseline[f] for f in FOLDS])
    for trial in trials:trial['keep']=all(trial['metrics'][f]['wer']<=baseline[f]['wer']+.002 for f in FOLDS) and trial['metrics']['aggregate']['wer']<baseline['aggregate']['wer']-.001
    best=min(trials,key=lambda t:t['metrics']['aggregate']['wer']);m=best['metrics']
    write(OUT/'nbest_summary.json',{'report':f"Cached Whisper candidate-set bigram rescoring: A {m['fold_A']['wer']:.6f}, B {m['fold_B']['wer']:.6f}, aggregate {m['aggregate']['wer']:.6f}; keep={best['keep']}.",
        'best':best,'trials':trials,'baseline':baseline,'candidate_source':'Actual top-3 beam5 hypotheses where generated, plus cached greedy/beam3/beam5/auto outputs on acoustically selected short clips; long clips retain greedy.',
        'training_text':'Exact conversation-disjoint competition training manifest plus Jember training text only; no validation/holdout/test references.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',choices=['soup','large_soup','adaptive','nbest'],required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    if a.step in ['soup','large_soup']:soup(a.deadline,a.step=='large_soup')
    elif a.step=='adaptive':adaptive(a.deadline)
    else:nbest(a.deadline)
