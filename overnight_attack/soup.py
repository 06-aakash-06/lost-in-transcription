"""Exact weighted averaging of compatible LoRA deltas, including cross-run."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file

def frozen_base_dtype(adapter):
    provenance=adapter/'soup_provenance.json'
    if provenance.exists():
        sources=json.loads(provenance.read_text())
        left=frozen_base_dtype(Path(sources['left']))
        right=frozen_base_dtype(Path(sources['right']))
        if left!=right:raise ValueError('Soup sources used different frozen-base dtypes')
        return left
    run_config=adapter.parent/'config.json'
    if run_config.exists():return json.loads(run_config.read_text()).get('dtype','float32')
    return 'float32'

def average(left,right,output,weight=.5):
    assert 0<weight<1
    dtype=frozen_base_dtype(left)
    assert dtype==frozen_base_dtype(right),'incompatible frozen-base precision'
    if output.exists():
        previous=json.loads((output/'soup_provenance.json').read_text())
        assert previous['left']==str(left.resolve()) and previous['right']==str(right.resolve()) and previous['left_weight']==weight
        return output
    a=json.loads((left/'adapter_config.json').read_text());b=json.loads((right/'adapter_config.json').read_text())
    keys=['base_model_name_or_path','r','lora_alpha','lora_dropout','bias','fan_in_fan_out','use_rslora','use_dora','rank_pattern','alpha_pattern','modules_to_save','peft_type']
    assert all(a.get(k)==b.get(k) for k in keys),'incompatible LoRA base or layout'
    ta,tb=a['target_modules'],b['target_modules']
    assert ta==tb if isinstance(ta,str) else set(ta)==set(tb)
    assert a['bias']=='none' and not any(a.get(k) for k in ['use_rslora','use_dora','rank_pattern','alpha_pattern','modules_to_save'])
    base=Path(a['base_model_name_or_path']);assert base.is_dir()
    base_config_hash=hashlib.sha256((base/'config.json').read_bytes()).hexdigest()
    first=load_file(str(left/'adapter_model.safetensors'));second=load_file(str(right/'adapter_model.safetensors'))
    assert first.keys()==second.keys()
    result={}
    for name,tensor in first.items():
        assert tensor.shape==second[name].shape
        if '.lora_A.' in name:result[name]=torch.cat([tensor,second[name]],dim=0).contiguous()
        elif '.lora_B.' in name:result[name]=torch.cat([tensor*weight,second[name]*(1-weight)],dim=1).contiguous()
        else:raise ValueError(name)
    for name in result:
        if '.lora_A.' not in name:continue
        bn=name.replace('.lora_A.','.lora_B.');vector=torch.randn(first[name].shape[1],2,generator=torch.Generator().manual_seed(1337),dtype=first[name].dtype)
        expected=weight*(first[bn]@(first[name]@vector))+(1-weight)*(second[bn]@(second[name]@vector))
        actual=result[bn]@(result[name]@vector)
        assert torch.allclose(expected,actual,atol=3e-5,rtol=3e-5),('delta equivalence failure',name)
    output.mkdir(parents=True)
    for path in left.iterdir():
        if path.is_file() and path.suffix in ['.json','.txt'] and path.name not in ['train_metrics.json','adapter_config.json','valid.json','jember.json','soup_provenance.json']:
            shutil.copyfile(path,output/path.name)
    a['r']*=2;a['lora_alpha']*=2
    (output/'adapter_config.json').write_text(json.dumps(a,indent=2)+'\n')
    save_file(result,str(output/'adapter_model.safetensors'))
    hashes=[]
    for path in [left,right]:
        with (path/'adapter_model.safetensors').open('rb') as stream:hashes.append(hashlib.file_digest(stream,'sha256').hexdigest())
    provenance={'left':str(left.resolve()),'right':str(right.resolve()),'left_weight':weight,'source_sha256':hashes,
        'frozen_training_base_dtype':dtype,
        'base_config_sha256':base_config_hash,'method':'exact weighted LoRA deltas by factor concatenation','equivalence_test':'Fixed random-vector action for every adapted module; r/alpha scaling preserved.'}
    (output/'soup_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance),flush=True);return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--weight',type=float,default=.5);a=p.parse_args();average(a.left,a.right,a.output,a.weight)
