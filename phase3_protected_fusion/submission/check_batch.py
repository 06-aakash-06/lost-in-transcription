"""Check mixed-length MERaLiON batches on the exact extracted final model."""
import os,sys,json,socket,time
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
source=HERE/'results/offline_zip_tests/candidate_B/extracted';sys.path.insert(0,str(source))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HOME']=str(HERE/'results/batch_empty_cache');os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
def blocked(*a,**k):raise RuntimeError('NETWORK BLOCKED')
socket.socket.connect=blocked;socket.create_connection=blocked
import torch
from runtime.meralion import Meralion
from runtime.audio import load_audio
start=time.monotonic();torch.mps.set_per_process_memory_fraction(.9)
model=Meralion(source/'models/meralion','mps')
data=source.parent/'data/clips';waves=[load_audio(data/'long.mp3'),load_audio(data/'short,"quoted".mp3')]
single=[model.transcribe_arrays([w])[0] for w in waves]
batch=model.transcribe_arrays(waves)
result={'same_outputs':single==batch,'single':single,'batch':batch,'batch_size':2,'mixed_native_chunks':[2,1],'elapsed_s':time.monotonic()-start,'device':'mps'}
(HERE/'results/batch_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['single','batch']}));assert single==batch,'MERaLiON batch predictions differ'
