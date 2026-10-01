"""Resumable fresh zero-shot / adapted fold inference on the available device."""
from __future__ import annotations
import argparse,gc,json,sys,time
import torch
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase2_meralion.inference.meralion import BASE,DECODE,load,load_audio,transcribe
from phase1_whisper_anchor.training.metrics import corpus


def evaluate(manifest,output,adapter=None,base=BASE,limit=None):
    frame=pd.read_csv(manifest,sep='\t',keep_default_na=False)
    if limit:frame=frame.head(limit)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    previous_metrics=json.loads(output.with_suffix('.json').read_text()) if output.with_suffix('.json').exists() else {}
    rows=pd.read_csv(output,keep_default_na=False).to_dict('records') if output.exists() else []
    if len({r['clip_id'] for r in rows})!=len(rows):raise ValueError('duplicate prediction IDs')
    done={r['clip_id'] for r in rows};remaining=frame[~frame.clip_id.isin(done)]
    started=time.monotonic()
    merge_validation=None
    checked={}
    if len(remaining):
        model,processor,device,dtype=load(base,adapter=adapter);model.eval()
        print(f'device={device} dtype={dtype} remaining={len(remaining)}',flush=True)
        if adapter:
            samples=list(remaining.head(2).itertuples())
            waves=[load_audio(ROOT/item.path) for item in samples]
            original=[transcribe(model,processor,[wave],device,dtype)[0] for wave in waves]
            model.text_decoder.model.merge_adapter(safe_merge=True)
            merged=[transcribe(model,processor,[wave],device,dtype)[0] for wave in waves]
            matches=original==merged
            merge_validation={'samples':len(samples),'same_transcripts':matches,'mode':'merged_adapter' if matches else 'original_adapter'}
            if not matches:
                del model;gc.collect()
                if device=='mps':torch.mps.empty_cache()
                elif device=='cuda':torch.cuda.empty_cache()
                model,processor,device,dtype=load(base,adapter=adapter);model.eval()
            checked={item.clip_id:text for item,text in zip(samples,original)}
            print('evaluation merge validation',merge_validation,flush=True)
        for index,item in enumerate(remaining.itertuples()):
            text=checked[item.clip_id] if item.clip_id in checked else transcribe(model,processor,[load_audio(ROOT/item.path)],device,dtype)[0]
            rows.append({'clip_id':item.clip_id,'reference':item.text,'transcript':text})
            pd.DataFrame(rows).to_csv(output,index=False)
            if (index+1)%10==0:print(f'completed={len(rows)}/{len(frame)} elapsed={time.monotonic()-started:.1f}',flush=True)
    preds=pd.DataFrame(rows).set_index('clip_id').loc[frame.clip_id]
    if preds.reference.tolist()!=frame.text.tolist():raise ValueError('prediction reference mismatch')
    metrics=corpus(frame.text.tolist(),preds.transcript.tolist())
    elapsed=time.monotonic()-started if len(remaining) else previous_metrics.get('elapsed_this_invocation_s',0)
    metrics.update({'clips':len(frame),'elapsed_this_invocation_s':elapsed,'model':str(base),'adapter':str(adapter) if adapter else None,'decode':DECODE,'merge_eval_validation':merge_validation if len(remaining) else previous_metrics.get('merge_eval_validation')})
    output.with_suffix('.json').write_text(json.dumps(metrics,indent=2)+'\n');print(metrics,flush=True)
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--adapter',type=Path);p.add_argument('--base',type=Path,default=BASE);p.add_argument('--limit',type=int);a=p.parse_args();evaluate(a.manifest,a.output,a.adapter,a.base,a.limit)
