"""One fixed low-confidence N-best experiment, strictly conversation-disjoint."""
import sys,json,time,argparse,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import torch
from overnight_attack.evaluate import Engine,write
from overnight_attack.evidence import score,aggregate
from final_9h_attack.compare_fusion import frame,consensus
from final_9h_attack.runtime.repetition_guard import guard
from final_9h_attack.runtime.ngram import train,LM,eligible,generate,choose
from final_9h_attack.runtime.audio import load_audio
from phase3_protected_fusion.final_conservative_check import edit_stats
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def evaluate():
    (OUT/'nbest').mkdir(exist_ok=True)
    records={}
    for fold in ['fold_B','fold_A']:
        manifest=ROOT/f'phase1_whisper_anchor/data/{fold}_train.tsv';training=pd.read_csv(manifest,sep='\t',keep_default_na=False)
        lm_data=train(training.text);lm=LM(lm_data);write(OUT/f'nbest/{fold}_lm.json',lm_data)
        engine=None
        for dataset in ['dev','jember']:
            data=frame(fold,1.,dataset);confidence=json.loads((OUT/f'confidence/1.0/{dataset}_{fold}.json').read_text())
            assert not set(data.clip_id)&set(training.clip_id),'LM training/evaluation overlap'
            path=OUT/f'nbest/{dataset}_{fold}.cache.json';cache=json.loads(path.read_text()) if path.exists() else {}
            outputs=[];baseline=[];queried=0;changed=0
            for index,row in enumerate(data.itertuples(),1):
                safe=guard(row.large_raw)
                core=safe if safe!=row.large_raw else consensus(row.large,row.turbo,row.meralion,0);baseline.append(core)
                conf=confidence[row.clip_id];assert conf['transcript']==row.large
                if safe==row.large_raw and eligible(row.large,core,conf['anchor'],row.duration_s):
                    queried+=1
                    if row.clip_id not in cache:
                        if engine is None:engine=Engine(ROOT/'overnight_attack/models/large_v3_base',ROOT/f'overnight_attack/runs/large_v3_{fold}/epoch_1_full',dtype='bfloat16',language='indonesian',beams=1)
                        candidates,acoustic=generate(engine,load_audio(ROOT/row.path),row.large)
                        cache[row.clip_id]={'anchor':row.large,'candidates':candidates,'scores':acoustic};write(path,cache)
                    record=cache[row.clip_id];assert record['anchor']==row.large
                    selected=choose(row.large,record['candidates'],record['scores'],lm)
                    cleaned=guard(selected)
                    output=cleaned if cleaned!=selected else consensus(selected,row.turbo,row.meralion,0)
                    changed+=output!=core
                else:output=core
                outputs.append(output)
                if index%20==0:print('NBEST',dataset,fold,index,len(data),'queried',queried,'changed',changed,flush=True)
            metrics=score(data.text.tolist(),outputs);stats=[edit_stats(a,b) for a,b in zip(data.large,outputs)]
            metrics.update(queried_clips=queried,changed_vs_C_clips=changed,edits={k:sum(e[k] for e in stats) for k in stats[0]})
            records.setdefault(dataset,{})[fold]={'candidate':metrics,'baseline_C':score(data.text.tolist(),baseline)}
            pd.DataFrame({'clip_id':data.clip_id,'reference':data.text,'transcript':outputs}).to_csv(OUT/f'nbest/{dataset}_{fold}.csv',index=False)
        if engine is not None:del engine
        gc.collect();torch.mps.empty_cache()
    for dataset in records:
        records[dataset]['aggregate']={name:aggregate([records[dataset][f][name] for f in ['fold_A','fold_B']]) for name in ['candidate','baseline_C']}
    dev=records['dev'];j=records['jember']
    passed=all(dev[f]['candidate']['wer']<dev[f]['baseline_C']['wer'] for f in ['fold_A','fold_B']) and dev['aggregate']['candidate']['wer']<=dev['aggregate']['baseline_C']['wer']-.0015 and j['aggregate']['candidate']['wer']<=j['aggregate']['baseline_C']['wer']+.001
    output={'recipe':'Only <=30s clips where C proposes edits and Large mean word probability <.8; top3 beam3 plus greedy, acoustic+.1*trigram, .02 acceptance margin, length within15%, no increased repetition; one fixed rule','results':records,'passes_continue_gate':passed,'training':'Each fold LM uses only that fold training manifest; Jember holdout excluded; no identifiers or references in selection.'}
    write(OUT/'nbest_comparison.json',output);print(json.dumps(output,indent=2),flush=True)
if __name__=='__main__':evaluate()
