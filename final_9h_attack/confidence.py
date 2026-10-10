"""Restartable acoustic scores, using hypotheses only, for supported edits."""
import os,sys,json,argparse,gc,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from overnight_attack.evaluate import Engine,write
from final_9h_attack.compare_fusion import frame,predict,CONFIGS,consensus
from final_9h_attack.decode_variants import checkpoint
from phase3_protected_fusion.runtime.confidence import score_pair
from phase3_protected_fusion.runtime.audio import load_audio
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def extract(coefficient):
    selected=json.loads((OUT/'anchor_selection.json').read_text())
    EngineClass=Engine
    if selected.get('prediction_mode')=='merged':
        from final_9h_attack.evaluate_merged import MergedEngine
        EngineClass=MergedEngine
    for fold in ['fold_A','fold_B']:
        engine=None
        for dataset in ['dev','jember']:
            data=frame(fold,coefficient,dataset);path=OUT/f'confidence/{coefficient}/{dataset}_{fold}.json';path.parent.mkdir(parents=True,exist_ok=True)
            cache=json.loads(path.read_text()) if path.exists() else {};started=time.monotonic()
            for index,row in enumerate(data.itertuples(),1):
                if row.clip_id in cache:
                    assert cache[row.clip_id]['transcript']==row.large;continue
                proposed=consensus(row.large,row.turbo,row.meralion,0)!=row.large or predict(row,CONFIGS['A Turbo+MER strict'])!=row.large
                if row.duration_s>30 or row.large_raw!=row.large or not proposed:values=[None]*len(row.large.split())
                else:
                    if engine is None:engine=EngineClass(ROOT/'overnight_attack/models/large_v3_base',checkpoint(fold,coefficient),dtype='bfloat16',language=selected['language'],beams=selected['beams'])
                    values=score_pair(engine,load_audio(ROOT/row.path),[row.large])[0]
                    assert len(values)==len(row.large.split())
                cache[row.clip_id]={'transcript':row.large,'anchor':values}
                if index%10==0:write(path,cache);print('CONFIDENCE',dataset,fold,index,len(data),'seconds',round(time.monotonic()-started,1),flush=True)
            assert set(cache)==set(data.clip_id);write(path,cache)
        if engine is not None:del engine
        gc.collect()
        if torch.backends.mps.is_available():torch.mps.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--coefficient',type=float,required=True);a=p.parse_args();extract(a.coefficient)
