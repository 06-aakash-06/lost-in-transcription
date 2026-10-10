"""Real-model logits/tokens equivalence against PEFT, including native long audio."""
import sys,argparse,json,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,pandas as pd
from safetensors.torch import load_file
from overnight_attack.evaluate import Engine
from final_9h_attack.runtime.delta import inject
from phase1_whisper_anchor.inference.audio import load_audio

def prove(adapter,output):
    engine=Engine(ROOT/'overnight_attack/models/large_v3_base',adapter,dtype='bfloat16',batch_size=1)
    frame=pd.read_csv(ROOT/'phase1_whisper_anchor/data/fold_A_valid.tsv',sep='\t',keep_default_na=False)
    ids=['71ac896ea0bd4999ac7dad89c7298da1','3fc1b82ff2e64625af2a92801b63cc8f'];rows=frame.set_index('clip_id').loc[ids]
    waves=[load_audio(ROOT/p) for p in rows.path]
    features=engine.processor.feature_extractor(waves[0],sampling_rate=16000,return_tensors='pt').input_features.to(engine.device,dtype=engine.dtype)
    decoder=torch.tensor([[50258,50275,50360,50364]],device=engine.device)
    with torch.inference_mode():before=engine.model(input_features=features,decoder_input_ids=decoder).logits.cpu()
    texts_before=engine.transcribe_arrays(waves)
    model=engine.model.get_base_model();config=json.loads((adapter/'adapter_config.json').read_text())
    weights=load_file(str(adapter/'adapter_model.safetensors'),device='cpu')
    count=inject(model,weights,config['lora_alpha']/config['r']);engine.model=model
    with torch.inference_mode():after=engine.model(input_features=features,decoder_input_ids=decoder).logits.cpu()
    texts_after=engine.transcribe_arrays(waves)
    difference=float((before-after).abs().max())
    result={'device':engine.device,'base_dtype':'bfloat16','delta_dtype':'float32','adapted_modules':count,'max_abs_logit_difference':difference,'same_normal_and_native_long_transcripts':texts_before==texts_after,'normal_and_long_durations':rows.duration_s.tolist(),'method':'Reloaded saved safetensor deltas and replaced every PEFT linear with inference-only two-projection arithmetic; no merge/cast loss.','pass':difference<1e-5 and texts_before==texts_after}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not result['pass']:raise RuntimeError('Delta runtime differs from validated PEFT model')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prove(a.adapter,a.output)
