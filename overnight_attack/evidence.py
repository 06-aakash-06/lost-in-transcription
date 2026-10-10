"""Cached OOF metrics, clip-level oracle bounds, and explicit Large-v3 gates."""
import argparse
import json
import sys
from pathlib import Path
from functools import lru_cache
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase1_whisper_anchor.training.metrics import counts
from phase3_protected_fusion.runtime.fusion import collapse_single_runs
from phase4_stronger_anchor.audit import aligned
HERE=ROOT/'overnight_attack';OUT=HERE/'results'

@lru_cache(maxsize=30000)
def error(ref,hyp):return counts(ref,hyp)

def score(refs,hyps):
    n=s=d=i=0
    for r,h in zip(refs,hyps):
        a,b,c,e=error(r,h);n+=a;s+=b;d+=c;i+=e
    return {'wer':(s+d+i)/n,'words':n,'substitutions':s,'deletions':d,'insertions':i}

def aggregate(records):
    total={k:sum(r[k] for r in records) for k in ['words','substitutions','deletions','insertions']}
    total['wer']=sum(total[k] for k in ['substitutions','deletions','insertions'])/total['words'];return total

def oracle(refs,*systems):
    selected=[min(options,key=lambda h:sum(error(ref,h)[1:])) for ref,options in zip(refs,zip(*systems))]
    return score(refs,selected)

def fold_evidence(fold,run):
    manifest=pd.read_csv(ROOT/f'phase1_whisper_anchor/data/{fold}_valid.tsv',sep='\t',keep_default_na=False)
    held=pd.read_csv(ROOT/'phase1_whisper_anchor/data/jember_holdout.tsv',sep='\t',keep_default_na=False)
    refs=manifest.text.tolist();h_refs=held.text.tolist()
    turbo=list(map(collapse_single_runs,aligned(ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/epoch_1_full/valid.csv',manifest)))
    r4=list(map(collapse_single_runs,aligned(ROOT/f'phase1_whisper_anchor/runs/indonesian_x4_{fold}/epoch_1_full/valid.csv',manifest)))
    meralion=aligned(ROOT/f'phase2_meralion/results/pred_epoch_1_half_{fold}.csv',manifest)
    t_held=list(map(collapse_single_runs,aligned(ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/epoch_1_full/jember.csv',held)))
    summary={'fold':fold,'run':run,'Turbo_R3_full_safe':score(refs,turbo),'Turbo_R4_full_safe':score(refs,r4),
        'MERaLiON':score(refs,meralion),'Turbo_Jember':score(h_refs,t_held),'checkpoints':{}}
    for checkpoint in ['epoch_1_half','epoch_1_full']:
        directory=OUT/'predictions'/run/checkpoint
        if not (directory/f'{fold}_valid.csv').exists():continue
        raw=aligned(directory/f'{fold}_valid.csv',manifest);large=list(map(collapse_single_runs,raw))
        value={'raw':score(refs,raw),'safe':score(refs,large),'pair_oracle_Turbo_R3':oracle(refs,large,turbo),
            'pair_oracle_Turbo_R4':oracle(refs,large,r4),'pair_oracle_MERaLiON':oracle(refs,large,meralion),
            'triple_oracle':oracle(refs,large,turbo,meralion)}
        if (directory/'jember_holdout.csv').exists():
            h=list(map(collapse_single_runs,aligned(directory/'jember_holdout.csv',held)));value['jember']=score(h_refs,h)
        baseline=summary['Turbo_R3_full_safe']['wer'];new=value['safe']['wer'];oracle_gain=baseline-value['pair_oracle_Turbo_R3']['wer']
        value['gate']={'beats_Turbo_by_005':new<=baseline-.005,'near_Turbo_and_oracle_gain_015':new<=baseline+.005 and oracle_gain>=.015,
            'OOD_gain_without_fold_collapse':value.get('jember',{}).get('wer',1)<=summary['Turbo_Jember']['wer']-.015 and new<=baseline+.01}
        value['passes_continue_gate']=any(value['gate'].values());summary['checkpoints'][checkpoint]=value
    summary['passes_continue_gate']=any(v['passes_continue_gate'] for v in summary['checkpoints'].values())
    path=OUT/f'large_v3_{fold}_evidence.json';path.write_text(json.dumps(summary,indent=2)+'\n')
    return summary

def two_fold():
    summaries=[json.loads((OUT/f'large_v3_{fold}_evidence.json').read_text()) for fold in ['fold_A','fold_B']]
    candidates={}
    for checkpoint in ['epoch_1_half','epoch_1_full']:
        if not all(checkpoint in s['checkpoints'] for s in summaries):continue
        records=[s['checkpoints'][checkpoint] for s in summaries]
        combined={key:aggregate([r[key] for r in records]) for key in ['raw','safe','pair_oracle_Turbo_R3','pair_oracle_Turbo_R4','pair_oracle_MERaLiON','triple_oracle','jember']}
        combined['worst_fold']=max(r['safe']['wer'] for r in records)
        combined['fold_A']=records[0]['safe'];combined['fold_B']=records[1]['safe']
        # Independent usefulness on both conversations is required for final
        # training; no final all-data model is evaluated against seen clips.
        combined['both_fold_gate']=all(r['passes_continue_gate'] for r in records)
        candidates[checkpoint]=combined
    eligible=[c for c,v in candidates.items() if v['both_fold_gate']]
    selected=min(eligible or list(candidates),key=lambda c:(candidates[c]['safe']['wer'],candidates[c]['worst_fold']))
    data={'selected_checkpoint':selected,'candidates':candidates,'passes_final_training_gate':bool(eligible),
        'Turbo_aggregate':aggregate([s['Turbo_R3_full_safe'] for s in summaries]),'oracle_note':'Reference-selected whole-clip hypotheses; an analysis lower bound, not deployed selection.'}
    (OUT/'large_v3_two_fold.json').write_text(json.dumps(data,indent=2)+'\n');return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['fold_A','fold_B']);p.add_argument('--run');p.add_argument('--two-fold',action='store_true');a=p.parse_args()
    print(json.dumps(two_fold() if a.two_fold else fold_evidence(a.fold,a.run),indent=2),flush=True)
