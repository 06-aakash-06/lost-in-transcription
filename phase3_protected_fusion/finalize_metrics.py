"""Verify selected runtime configs, oracle headroom and correction diagnostics."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase3_protected_fusion.data import load
from phase3_protected_fusion.runtime.fusion import fuse
from phase1_whisper_anchor.training.metrics import counts,corpus
import pandas as pd
HERE=Path(__file__).resolve().parent
summary={};allrefs=[];allpred={k:[] for k in ['whisper','meralion','pair_oracle','ratio4_oracle','auto_oracle','full_oracle','candidate_A','candidate_B']};diagnostics=[]
for fold in ['fold_A','fold_B']:
    frame=load(fold);c=json.loads((HERE/f'results/confidence_{fold}.json').read_text());pred={k:[] for k in allpred}
    configs={k:json.loads((HERE/f'configs/{k}.json').read_text())['fusion'] for k in ['candidate_A','candidate_B']}
    for x in frame.itertuples():
        choices={'whisper':x.anchor,'meralion':x.meralion,'ratio4':x.ratio4,'auto':x.auto,'full':x.full}
        errors={k:sum(counts(x.text,v)[1:]) for k,v in choices.items()}
        pred['whisper'].append(x.anchor);pred['meralion'].append(x.meralion)
        for voter in ['pair','ratio4','auto','full']:
            options=['whisper','meralion']+([] if voter=='pair' else [voter]);best=min(options,key=lambda k:errors[k]);pred[voter+'_oracle'].append(choices[best])
        for candidate,cfg in configs.items():
            # Same runtime evidence: only anchor scores, no reference or MER scores.
            text=fuse(x.anchor,x.meralion,cfg,{'anchor':c[x.clip_id]['anchor']},x.full if candidate=='candidate_B' else None)
            pred[candidate].append(text)
            delta=errors['whisper']-sum(counts(x.text,text)[1:])
            diagnostics.append({'fold':fold,'clip_id':x.clip_id,'candidate':candidate,'changed':text!=x.anchor,'error_gain':delta,'whisper_errors':errors['whisper'],'meralion_errors':errors['meralion']})
    summary[fold]={k:corpus(frame.text.tolist(),v) for k,v in pred.items()}
    for k in allpred:allpred[k].extend(pred[k])
    allrefs.extend(frame.text.tolist())
    for candidate in configs:
        pd.DataFrame({'clip_id':frame.clip_id,'reference':frame.text,'transcript':pred[candidate]}).to_csv(HERE/f'results/{candidate}_{fold}.csv',index=False)
summary['aggregate']={k:corpus(allrefs,v) for k,v in allpred.items()}
(HERE/'results/final_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
pd.DataFrame(diagnostics).to_csv(HERE/'results/correction_diagnostics.csv',index=False)
for f,v in summary.items():print(f,{k:round(m['wer'],6) for k,m in v.items()})
