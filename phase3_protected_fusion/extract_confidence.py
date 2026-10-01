import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase3_protected_fusion.data import load
from phase3_protected_fusion.runtime.confidence import score_pair
from phase1_whisper_anchor.inference.whisper import WhisperAnchor
from phase1_whisper_anchor.inference.audio import load_audio
HERE=Path(__file__).resolve().parent

def run(fold):
    output=HERE/f'results/confidence_{fold}.json'
    done=json.loads(output.read_text()) if output.exists() else {}
    frame=load(fold)
    remaining=frame[~frame.clip_id.isin(done)]
    if not len(remaining):return
    engine=WhisperAnchor.from_adapter(str(ROOT/'submission_src/models/whisper_large_v3_turbo'),ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/epoch_1_half','indonesian',1)
    started=time.monotonic()
    for item in remaining.itertuples():
        a,m=score_pair(engine,load_audio(ROOT/item.path),[item.anchor,item.meralion])
        assert len(a)==len(item.anchor.split()) and len(m)==len(item.meralion.split())
        done[item.clip_id]={'anchor':a,'meralion':m}
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(done)+'\n');tmp.replace(output)
        if len(done)%20==0:print(f'{fold} scored={len(done)}/{len(frame)} elapsed={time.monotonic()-started:.1f}',flush=True)
    print(f'{fold} COMPLETE elapsed={time.monotonic()-started:.1f}',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['fold_A','fold_B'],required=True);args=p.parse_args();run(args.fold)
