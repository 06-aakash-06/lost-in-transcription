"""Restartable local diagnostic using the exact standalone runtime engine."""
import os,sys,json,time,argparse,hashlib
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import torch
from final_9h_attack.runtime.whisper import Whisper
from final_9h_attack.runtime.audio import load_audio
from final_9h_attack.runtime.fusion import collapse_single_runs
from overnight_attack.evidence import score
from overnight_attack.evaluate import write

def evaluate(model,config,manifest,output,batch_size=4):
    frame=pd.read_csv(manifest,sep='\t',keep_default_na=False);assert frame.clip_id.is_unique
    hashes={}
    for p in sorted(model.glob('*.safetensors')):
        with p.open('rb') as stream:hashes[p.name]=hashlib.file_digest(stream,'sha256').hexdigest()
    assert hashes
    provenance={'model':str(model.resolve()),'weights':hashes,'config':config,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'batch_size':batch_size,'mode':'exact standalone submission engine'}
    output.parent.mkdir(parents=True,exist_ok=True);cache_path=output.with_suffix('.cache.json')
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {'provenance':provenance,'predictions':{},'elapsed_s':0}
    assert cache['provenance']==provenance
    for r in frame.itertuples():
        if r.clip_id in cache['predictions']:assert cache['predictions'][r.clip_id]['reference']==r.text
    remaining=frame[~frame.clip_id.isin(cache['predictions'])]
    if len(remaining):
        torch.manual_seed(1337);engine=Whisper(model,config,batch_size);started=time.monotonic();previous=cache['elapsed_s']
        for start in range(0,len(remaining),batch_size):
            part=remaining.iloc[start:start+batch_size];waves=[load_audio(ROOT/p) for p in part.path];texts=engine.transcribe_arrays(waves)
            assert len(texts)==len(part)
            for r,t in zip(part.itertuples(),texts):cache['predictions'][r.clip_id]={'reference':r.text,'transcript':t}
            cache['elapsed_s']=previous+time.monotonic()-started;write(cache_path,cache)
            if start%(5*batch_size)==0:print(output.stem,len(cache['predictions']),len(frame),round(cache['elapsed_s'],1),flush=True)
    texts=[cache['predictions'][c]['transcript'] for c in frame.clip_id]
    result={'provenance':provenance,'raw':score(frame.text.tolist(),texts),'safe':score(frame.text.tolist(),list(map(collapse_single_runs,texts))),'elapsed_s':cache['elapsed_s']}
    pd.DataFrame({'clip_id':frame.clip_id,'reference':frame.text,'transcript':texts}).to_csv(output.with_suffix('.csv'),index=False);write(output,result)
    print('COMPLETE',output,json.dumps(result['safe']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args();evaluate(a.model,json.loads(a.config.read_text()),a.manifest,a.output,a.batch_size)
