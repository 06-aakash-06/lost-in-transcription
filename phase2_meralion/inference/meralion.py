"""Local-only MERaLiON, frozen encoders and greedy transcription."""
from __future__ import annotations
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import sys
from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from phase1_whisper_anchor.inference.audio import RATE, load_audio
from phase1_whisper_anchor.inference.postprocess import clean
BASE=ROOT/'submission_src/models/meralion3_asr'
DECODE={'max_new_tokens':440,'do_sample':False,'no_repeat_ngram_size':0,'repetition_penalty':1.05}
PROMPT='Instruction: Please transcribe this speech. \nFollow the text instruction based on the following audio: <SpeechHere>'


def device_dtype(training=False):
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device=='mps' and hasattr(torch.mps,'set_per_process_memory_fraction'):
        try:torch.mps.set_per_process_memory_fraction(0.9)
        except (RuntimeError,NotImplementedError):pass
    # Gemma2 softcapping is safer in bfloat16 than float16. Mac supports BF16.
    dtype=torch.bfloat16 if device=='cuda' or (device=='mps' and training) else torch.float16 if device=='mps' else torch.float32
    return device,dtype


def load(base=BASE,training=False,adapter=None):
    device,dtype=device_dtype(training)
    processor=AutoProcessor.from_pretrained(str(base),local_files_only=True,trust_remote_code=True)
    model=AutoModelForSpeechSeq2Seq.from_pretrained(str(base),local_files_only=True,trust_remote_code=True,dtype=dtype,attn_implementation='eager')
    if adapter:
        from peft import PeftModel
        model.text_decoder.model=PeftModel.from_pretrained(model.text_decoder.model,str(adapter),local_files_only=True)
    model=model.to(device)
    return model,processor,device,dtype


def prompt(processor):
    return processor.tokenizer.apply_chat_template([{'role':'user','content':PROMPT}],tokenize=False,add_generation_prompt=True)


def move(inputs,device,dtype):
    return {k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()}


@torch.inference_mode()
def transcribe(model,processor,waves,device,dtype):
    inputs=processor(text=[prompt(processor)]*len(waves),audios=waves,sampling_rate=RATE,return_tensors='pt',padding=True)
    ids=model.generate(**move(inputs,device,dtype),**DECODE)
    # Decoder-only generation includes its prompt. Slice IDs, not text.
    generated=ids[:,inputs['input_ids'].shape[1]:]
    return [clean(s) for s in processor.batch_decode(generated,skip_special_tokens=True)]
