"""Exact equal average of two compatible LoRA weight deltas, not factors.

Stack A factors and concatenate half-scaled B factors, keeping alpha/r fixed.
The resulting rank-64 adapter equals .5*(B1@A1 + B2@A2) before merging.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

def average(left, right, output):
    if left.parent.resolve() != right.parent.resolve(): raise ValueError('averaging is restricted to checkpoints from the same run')
    if output.exists(): raise FileExistsError(output)
    configs=[json.loads((path/'adapter_config.json').read_text()) for path in [left,right]]
    if configs[0] != configs[1]: raise ValueError('incompatible adapter configs')
    config=configs[0]
    if config.get('rank_pattern') or config.get('alpha_pattern') or config.get('use_rslora') or config.get('use_dora') or config.get('modules_to_save') or config.get('bias') != 'none':
        raise ValueError('unsupported averaging configuration')
    first=load_file(str(left/'adapter_model.safetensors'));second=load_file(str(right/'adapter_model.safetensors'))
    if first.keys() != second.keys(): raise ValueError('adapter keys differ')
    result={};checked=0
    for name, tensor in first.items():
        if tensor.shape != second[name].shape: raise ValueError('adapter shapes differ')
        if '.lora_A.' in name:
            result[name]=torch.cat([tensor,second[name]],dim=0).contiguous()
        elif '.lora_B.' in name:
            result[name]=torch.cat([tensor*.5,second[name]*.5],dim=1).contiguous()
        else: raise ValueError(f'unsupported adapter weight: {name}')
    for name in result:
        if '.lora_A.' not in name: continue
        bn=name.replace('.lora_A.','.lora_B.')
        # Test action on fixed random vectors, avoiding large dense matrices.
        generator=torch.Generator().manual_seed(1337)
        vector=torch.randn(first[name].shape[1],3,generator=generator)
        expected=.5*(first[bn]@(first[name]@vector)+second[bn]@(second[name]@vector))
        actual=result[bn]@(result[name]@vector)
        if not torch.allclose(expected,actual,atol=2e-5,rtol=2e-5): raise RuntimeError(f'averaging mismatch {name}')
        checked+=1
    config['r']*=2;config['lora_alpha']*=2
    output.mkdir(parents=True)
    for path in left.iterdir():
        if path.is_file() and path.suffix in ['.json','.txt'] and path.name not in ['train_metrics.json','adapter_config.json','valid.json','jember.json']:
            shutil.copyfile(path,output/path.name)
    (output/'adapter_config.json').write_text(json.dumps(config,indent=2)+'\n')
    save_file(result,str(output/'adapter_model.safetensors'))
    hashes=[]
    for path in [left,right]:
        with (path/'adapter_model.safetensors').open('rb') as stream:hashes.append(hashlib.file_digest(stream,'sha256').hexdigest())
    provenance={'source_checkpoints':[str(left),str(right)],'source_adapter_sha256':hashes,'method':'exact_equal_average_weight_deltas','verified_modules':checked}
    (output/'average_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    average(a.left,a.right,a.output)
