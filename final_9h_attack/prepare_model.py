"""Self-contained model folder; original model and adapter files stay intact."""
import argparse,json,hashlib,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from overnight_attack.soup import frozen_base_dtype

def prepare(adapter,output):
    base=ROOT/'overnight_attack/models/large_v3_base';config=json.loads((adapter/'adapter_config.json').read_text())
    assert config['bias']=='none' and not any(config.get(k) for k in ['use_dora','use_rslora','rank_pattern','alpha_pattern','modules_to_save','fan_in_fan_out'])
    metadata={'frozen_base_dtype':frozen_base_dtype(adapter),'r':config['r'],'alpha':config['lora_alpha'],'scale':config['lora_alpha']/config['r'],'model_id':'openai/whisper-large-v3','base_revision':'06f233fe06e710322aca913c1bc4249a0d71fce1','method':'Vanilla LoRA inference: BF16 frozen base plus FP32 residual projections. No PEFT dependency.'}
    with (adapter/'adapter_model.safetensors').open('rb') as stream:metadata['delta_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'lora_runtime.json').exists():assert json.loads((output/'lora_runtime.json').read_text())==metadata;return
    for p in base.iterdir():
        if p.is_file():
            target=output/p.name
            if p.suffix=='.safetensors':os.link(p,target)
            else:shutil.copyfile(p,target)
    processor_assets={'preprocessor_config.json','tokenizer_config.json','tokenizer.json','special_tokens_map.json','added_tokens.json','merges.txt','vocab.json','normalizer.json','generation_config.json'}
    for p in adapter.iterdir():
        if p.is_file() and p.name in processor_assets:
            shutil.copyfile(p,output/p.name)
    assert json.loads((output/'config.json').read_text())['decoder_layers']==32
    os.link(adapter/'adapter_model.safetensors',output/'domain_delta.safetensors')
    (output/'lora_runtime.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.adapter,a.output)
