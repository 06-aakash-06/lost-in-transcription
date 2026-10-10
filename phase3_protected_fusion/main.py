"""Offline inference only with independently decoded merged ASR models."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import gc,json,time
from pathlib import Path
import pandas as pd
import torch
from runtime.audio import load_audio
from runtime.whisper import WhisperAnchor
from runtime.meralion import Meralion
from runtime.confidence import score_pair
from runtime.fusion import fuse

def validate_output(metadata,output):
    if list(output.columns)!=['audio_filename','transcript']:raise ValueError('incorrect output columns')
    if len(output)!=len(metadata):raise ValueError('incorrect row count')
    if output.audio_filename.tolist()!=metadata.audio_filename.tolist():raise ValueError('filename order changed')
    if output.transcript.isna().any() or not output.transcript.map(lambda x:isinstance(x,str) and bool(x.strip())).all():raise ValueError('invalid or empty transcripts')

def select_device():
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def release(device):
    gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    elif device=='mps':torch.mps.empty_cache()

def run(data_dir,output_path,here):
    started=time.monotonic();print('starting inference',flush=True)
    config=json.loads((here/'config.json').read_text())
    metadata=pd.read_csv(data_dir/'test_metadata.csv',dtype={'audio_filename':str},keep_default_na=False)
    if 'audio_filename' not in metadata:raise ValueError('invalid metadata')
    names=metadata.audio_filename.tolist()
    if any(not isinstance(n,str) or not n or Path(n).name!=n for n in names):raise ValueError('invalid audio path')
    device=select_device()
    print('device='+device,flush=True)
    if device=='mps':torch.mps.set_per_process_memory_fraction(.9)
    batch_size=config['cuda_batch_size'] if device=='cuda' else 1
    clips=data_dir/'clips'
    anchor=WhisperAnchor(here/'models'/config.get('anchor_model','whisper'),language='indonesian',batch_size=batch_size,long_mode='native')
    print('Whisper loaded',flush=True)
    anchors=[];confidences=[]
    # One independent clip's scores never influence another clip.
    for start in range(0,len(names),batch_size):
        waves=[load_audio(clips/n,redact_errors=True) for n in names[start:start+batch_size]]
        texts=anchor.transcribe_arrays(waves)
        if len(texts)!=len(waves):raise ValueError('Whisper output count mismatch')
        anchors.extend(texts)
        for wave,text in zip(waves,texts):
            confidences.append({'anchor':score_pair(anchor,wave,[text])[0]} if config['fusion'].get('mode') in ('confidence','hybrid','support_confidence') else {})
    # Sequential residency also fits the local 16 GB Mac. CUDA batches retain
    # model throughput without allocating two sets of activation/cache memory.
    del anchor;release(device)
    third_texts=None
    if config.get('third_model'):
        voter=WhisperAnchor(here/'models'/config['third_model'],language='indonesian',batch_size=batch_size,long_mode='native')
        print('third voter loaded',flush=True);third_texts=[]
        for start in range(0,len(names),batch_size):
            waves=[load_audio(clips/n,redact_errors=True) for n in names[start:start+batch_size]]
            texts=voter.transcribe_arrays(waves)
            if len(texts)!=len(waves):raise ValueError('third voter output count mismatch')
            third_texts.extend(texts)
        del voter;release(device)
    corrective=Meralion(here/'models/meralion',device)
    print('MERaLiON loaded',flush=True)
    final=[]
    for start in range(0,len(names),batch_size):
        waves=[load_audio(clips/n,redact_errors=True) for n in names[start:start+batch_size]]
        hypotheses=corrective.transcribe_arrays(waves)
        for j,hyp in enumerate(hypotheses,start):final.append(fuse(anchors[j],hyp,config['fusion'],confidences[j],third_texts[j] if third_texts is not None else None))
    output=pd.DataFrame({'audio_filename':names,'transcript':final});validate_output(metadata,output)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=output_path.with_suffix('.tmp');output.to_csv(temporary,index=False);os.replace(temporary,output_path)
    print('inference complete',flush=True);print('submission written',flush=True);print(f'elapsed={time.monotonic()-started:.1f}',flush=True)

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    run(Path(os.environ.get('LIT_DATA_DIR','/code_execution/data')),Path(os.environ.get('LIT_SUBMISSION_PATH','/code_execution/submission/submission.csv')),here)
