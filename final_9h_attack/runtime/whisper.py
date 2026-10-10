"""Offline Whisper decoding, with exact saved deltas and native long-form."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import torch,json
from pathlib import Path
from transformers import WhisperProcessor,WhisperForConditionalGeneration
from .audio import RATE
from .postprocess import clean

class Whisper:
    def __init__(self,path,config,batch_size):
        self.device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.dtype=getattr(torch,config.get(self.device+'_dtype','float32'))
        if self.device=='cuda' and self.dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():self.dtype=torch.float32
        if self.device=='mps':torch.mps.set_per_process_memory_fraction(.88)
        self.processor=WhisperProcessor.from_pretrained(path,local_files_only=True)
        delta=Path(path)/'lora_runtime.json'
        base_dtype=getattr(torch,json.loads(delta.read_text())['frozen_base_dtype']) if delta.exists() else self.dtype
        self.model=WhisperForConditionalGeneration.from_pretrained(path,local_files_only=True,torch_dtype=base_dtype,attn_implementation='sdpa').to(device=self.device,dtype=self.dtype).eval()
        if delta.exists():
            from .delta import load
            load(self.model,Path(path))
        self.model.generation_config.language=None
        self.language=None if config.get('language','indonesian')=='auto' else config.get('language','indonesian')
        self.beams=config.get('beams',1);self.batch_size=batch_size

    @torch.inference_mode()
    def _decode(self,waves,long_form=False):
        if long_form:
            if len(waves)!=1:raise ValueError('long-form requires one independent clip')
            features=self.processor.feature_extractor(waves[0],sampling_rate=RATE,truncation=False,padding='longest',return_tensors='pt',return_attention_mask=True)
        else:features=self.processor.feature_extractor(waves,sampling_rate=RATE,return_tensors='pt',return_attention_mask=True)
        args={'num_beams':self.beams,'do_sample':False,'task':'transcribe','return_timestamps':long_form,'max_new_tokens':440}
        if self.language is not None:args['language']=self.language
        ids=self.model.generate(input_features=features.input_features.to(self.device,dtype=self.dtype),attention_mask=features.attention_mask.to(self.device),**args)
        return [clean(s) for s in self.processor.batch_decode(ids,skip_special_tokens=True)]

    def _safe(self,waves):
        try:return self._decode(waves)
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower() or len(waves)==1:raise
        # Leave the exception scope first so its traceback releases tensors.
        if self.device=='cuda':torch.cuda.empty_cache()
        elif self.device=='mps':torch.mps.empty_cache()
        middle=len(waves)//2;return self._safe(waves[:middle])+self._safe(waves[middle:])

    def transcribe_arrays(self,waves):
        result=['']*len(waves);short=[i for i,w in enumerate(waves) if len(w)<=30*RATE]
        for start in range(0,len(short),self.batch_size):
            indices=short[start:start+self.batch_size]
            for i,text in zip(indices,self._safe([waves[i] for i in indices])):result[i]=text
        for i,wave in enumerate(waves):
            if len(wave)>30*RATE:result[i]=self._decode([wave],True)[0]
        return result
