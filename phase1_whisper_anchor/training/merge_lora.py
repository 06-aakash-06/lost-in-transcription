"""Merge selected PEFT adapter and compare adapter versus saved merged logits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import sys
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from inference.audio import load_audio

def merge(base_path: str, adapter_path: Path, output: Path, sample_manifest: Path, language: str|None) -> dict:
    device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    processor=WhisperProcessor.from_pretrained(adapter_path,local_files_only=True)
    local_base=Path(base_path).exists()
    base=WhisperForConditionalGeneration.from_pretrained(base_path,torch_dtype=torch.float32,local_files_only=local_base).to(device)
    adapter=PeftModel.from_pretrained(base,adapter_path,local_files_only=True).to(device).eval()
    # Deterministic sample compares the exact same weights before and after save/load.
    features=torch.zeros((1,base.config.num_mel_bins,base.config.max_source_positions*2),dtype=torch.float32,device=device)
    labels=torch.tensor([[base.config.decoder_start_token_id,base.config.eos_token_id]],device=device)
    with torch.inference_mode(): before=adapter(input_features=features,decoder_input_ids=labels).logits
    sample=pd.read_csv(sample_manifest,sep="\t",keep_default_na=False).sort_values("duration_s").iloc[0]
    wave=load_audio(HERE.parent/sample.path)
    sample_features=processor.feature_extractor(wave,sampling_rate=16000,return_tensors="pt")
    input_features=sample_features.input_features.to(device)
    generation={"num_beams":1,"do_sample":False,"task":"transcribe","return_timestamps":False,"max_new_tokens":128}
    if language is not None: generation["language"]=language
    with torch.inference_mode(): before_ids=adapter.generate(input_features=input_features,**generation).cpu()
    merged=adapter.merge_and_unload().eval()
    output.mkdir(parents=True,exist_ok=True);merged.to("cpu").save_pretrained(output,safe_serialization=True);processor.save_pretrained(output)
    del adapter,base,merged
    if device=="mps": torch.mps.empty_cache()
    loaded=WhisperForConditionalGeneration.from_pretrained(output,local_files_only=True).to(device).eval()
    with torch.inference_mode(): after=loaded(input_features=features,decoder_input_ids=labels).logits
    diff=float((before-after).abs().max())
    with torch.inference_mode(): after_ids=loaded.generate(input_features=input_features,**generation).cpu()
    same_transcript=torch.equal(before_ids,after_ids)
    result={"max_abs_logit_difference":diff,"same_sample_token_ids":same_transcript,"equivalent":diff<1e-3 and same_transcript,"output":str(output),"device":device}
    (output/"merge_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    if not result["equivalent"]: raise RuntimeError(f"adapter/merged mismatch: {diff}")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--base-model",required=True);p.add_argument("--adapter",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--sample-manifest",type=Path,default=HERE/"data/fold_A_valid.tsv");p.add_argument("--language",choices=["indonesian","javanese","auto"],default="indonesian");a=p.parse_args();print(merge(a.base_model,a.adapter,a.output,a.sample_manifest,None if a.language=="auto" else a.language))
