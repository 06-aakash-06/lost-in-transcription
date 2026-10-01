"""Standalone merged MERaLiON inference, public custom code bundled locally."""
import torch
from transformers import AutoModelForSpeechSeq2Seq,AutoProcessor
from .postprocess import clean
from .audio import RATE
PROMPT='Instruction: Please transcribe this speech. \nFollow the text instruction based on the following audio: <SpeechHere>'

class Meralion:
    def __init__(self,path,device):
        self.device=device
        self.dtype=torch.bfloat16 if device=='cuda' else torch.float16 if device=='mps' else torch.float32
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True,trust_remote_code=True)
        self.processor.tokenizer.padding_side='left'
        self.model=AutoModelForSpeechSeq2Seq.from_pretrained(path,local_files_only=True,trust_remote_code=True,dtype=self.dtype,attn_implementation='eager').to(device).eval()
        self.prompt=self.processor.tokenizer.apply_chat_template([{'role':'user','content':PROMPT}],tokenize=False,add_generation_prompt=True)

    @torch.inference_mode()
    def transcribe_arrays(self,waves):
        inputs=self.processor(text=[self.prompt]*len(waves),audios=waves,sampling_rate=RATE,return_tensors='pt',padding=True)
        moved={k:v.to(device=self.device,dtype=self.dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
        ids=self.model.generate(**moved,max_new_tokens=440,do_sample=False,no_repeat_ngram_size=0,repetition_penalty=1.05)
        texts=self.processor.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        if len(texts)!=len(waves):raise ValueError('MERaLiON output count mismatch')
        return [clean(s) for s in texts]
