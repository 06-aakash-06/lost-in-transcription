"""Official WER and reference-only diagnostics; never used to route inference."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phase1_whisper_anchor.training.metrics import counts, corpus
from phase1_whisper_anchor.training.official_score import normalize_text
from submission_src.ensemble_runtime import Hypothesis, rover_consensus, _align_token_sequences


def correct_positions(ref, pred):
    aligned=_align_token_sequences(normalize_text(ref).split(),normalize_text(pred).split())
    return [a==b for a,b in aligned if a is not None]


def analyze(fold, meralion_csv, output):
    manifest=pd.read_csv(ROOT/f'phase1_whisper_anchor/data/{fold}_valid.tsv',sep='\t',keep_default_na=False)
    whisper=pd.read_csv(ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/epoch_1_half/valid.csv',keep_default_na=False).set_index('clip_id')
    meralion=pd.read_csv(meralion_csv,keep_default_na=False).set_index('clip_id')
    if not whisper.index.is_unique or not meralion.index.is_unique: raise ValueError('duplicate prediction IDs')
    if not set(manifest.clip_id).issubset(meralion.index): raise ValueError('incomplete MERaLiON predictions')
    rows=[]
    for item in manifest.itertuples():
        ref=item.text; w=whisper.loc[item.clip_id,'transcript']; m=meralion.loc[item.clip_id,'transcript']
        if whisper.loc[item.clip_id,'reference']!=ref: raise ValueError('reference mismatch')
        n,ws,wd,wi=counts(ref,w);_,ms,md,mi=counts(ref,m)
        we=ws+wd+wi;me=ms+md+mi
        fusion=rover_consensus([Hypothesis('whisper',w,'whisper'),Hypothesis('meralion',m,'meralion')],[1.,1.],{'order':['whisper','meralion'],'blank_factor':1.25})
        wc=correct_positions(ref,w);mc=correct_positions(ref,m)
        rows.append({'clip_id':item.clip_id,'reference':ref,'whisper':w,'meralion':m,'oracle':m if me<we else w,'fusion':fusion,'words':n,'whisper_errors':we,'meralion_errors':me,'meralion_recovers_ref_words':sum(not a and b for a,b in zip(wc,mc)),'whisper_recovers_ref_words':sum(a and not b for a,b in zip(wc,mc)),'both_miss_ref_words':sum(not a and not b for a,b in zip(wc,mc))})
    frame=pd.DataFrame(rows)
    metrics={name:corpus(frame.reference.tolist(),frame[name].tolist()) for name in ('whisper','meralion','oracle','fusion')}
    metrics.update({'fold':fold,'clips':len(frame),'oracle_gain':metrics['whisper']['wer']-metrics['oracle']['wer'],'whisper_wins':int(sum(frame.whisper_errors<frame.meralion_errors)),'meralion_wins':int(sum(frame.meralion_errors<frame.whisper_errors)),'ties':int(sum(frame.meralion_errors==frame.whisper_errors)),'both_fail':int(sum((frame.whisper_errors>0)&(frame.meralion_errors>0))),'meaningful_meralion_clip_fixes':int(sum(frame.whisper_errors-frame.meralion_errors>=2)),'meralion_recovers_ref_words':int(frame.meralion_recovers_ref_words.sum()),'whisper_recovers_ref_words':int(frame.whisper_recovers_ref_words.sum()),'both_miss_ref_words':int(frame.both_miss_ref_words.sum()),'fusion_method':'equal-weight two-system ROVER; substitution ties favor Whisper; blank_factor=1.25; diagnostic only'})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output.with_suffix('.csv'),index=False)
    output.with_suffix('.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2),flush=True)
    return metrics


def aggregate(metrics):
    result={}
    for name in ('whisper','meralion','oracle','fusion'):
        totals={key:sum(m[name][key] for m in metrics) for key in ('words','substitutions','deletions','insertions')}
        totals['wer']=sum(totals[k] for k in ('substitutions','deletions','insertions'))/totals['words']
        result[name]=totals
    result['oracle_gain']=result['whisper']['wer']-result['oracle']['wer']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['fold_A','fold_B'],required=True);p.add_argument('--meralion-csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.fold,a.meralion_csv,a.output)
