"""Offline, independent ASR; no test transcripts or identifiers are logged."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import gc,json,warnings
from pathlib import Path
import pandas as pd
import torch
from transformers.utils import logging
logging.set_verbosity_error()
logging.disable_progress_bar()
warnings.filterwarnings('ignore',category=FutureWarning)
from runtime.audio import load_audio
from runtime.whisper import Whisper
from runtime.fusion import fuse,collapse_single_runs
from runtime.consensus import consensus
from runtime.confidence import score_pair
from runtime.repetition_guard import guard

def select_device():
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def release(device):
    gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    elif device=='mps':torch.mps.empty_cache()

def validate_output(metadata,output):
    if list(output.columns)!=['audio_filename','transcript'] or len(output)!=len(metadata):raise ValueError('invalid output schema')
    if output.audio_filename.tolist()!=metadata.audio_filename.tolist():raise ValueError('invalid output order')
    if output.isna().any().any() or not output.transcript.map(lambda x:isinstance(x,str) and bool(x.strip())).all():raise ValueError('invalid output text')

def run(data_dir,output_path,here):
    config=json.loads((here/'config.json').read_text());device=select_device()
    torch.manual_seed(1337)
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=False
    metadata=pd.read_csv(data_dir/'test_metadata.csv',dtype={'audio_filename':str},keep_default_na=False)
    if 'audio_filename' not in metadata:raise ValueError('invalid input schema')
    names=metadata.audio_filename.tolist()
    if any(not n or Path(n).name!=n for n in names):raise ValueError('invalid input path')
    batch_size=config.get('cuda_batch_size',8) if device=='cuda' else 1
    texts={};confidences=[]
    for name,model in config['models'].items():
        if model['family']=='meralion':
            from runtime.meralion import Meralion
            engine=Meralion(here/'models'/name,device)
        else:engine=Whisper(here/'models'/name,model,batch_size)
        predictions=[]
        for start in range(0,len(names),batch_size):
            waves=[load_audio(data_dir/'clips'/n,redact_errors=True) for n in names[start:start+batch_size]]
            batch=engine.transcribe_arrays(waves)
            if len(batch)!=len(waves):raise ValueError('inference count mismatch')
            predictions.extend(batch)
            if name==config['anchor'] and (config.get('confidence_threshold') is not None or config.get('gate')):
                for wave,text in zip(waves,batch):confidences.append({'anchor':score_pair(engine,wave,[text])[0]})
        texts[name]=predictions;del engine;release(device)
    anchor=texts[config['anchor']];final=[]
    gate=config.get('gate')
    for i,text in enumerate(anchor):
        safe=guard(text) if config.get('phrase_guard',True) else collapse_single_runs(text)
        if config['strategy']=='standalone' or safe!=text:out=safe
        elif config['strategy']=='consensus':out=consensus(text,collapse_single_runs(texts['turbo'][i]),texts['meralion'][i],context=config.get('context',0),confidence=confidences[i] if confidences else None,threshold=config.get('confidence_threshold'),gate=gate)
        elif config['strategy']=='support':out=fuse(text,texts['meralion'][i],config['fusion'],third=collapse_single_runs(texts['turbo'][i]))
        else:raise ValueError('unsupported inference configuration')
        final.append(out)
    output=pd.DataFrame({'audio_filename':names,'transcript':final});validate_output(metadata,output)
    output_path.parent.mkdir(parents=True,exist_ok=True);temp=output_path.with_suffix('.tmp');output.to_csv(temp,index=False);os.replace(temp,output_path)

if __name__=='__main__':
    try:run(Path(os.environ.get('LIT_DATA_DIR','/code_execution/data')),Path(os.environ.get('LIT_SUBMISSION_PATH','/code_execution/submission/submission.csv')),Path(__file__).resolve().parent)
    except Exception:raise RuntimeError('submission inference failed') from None
