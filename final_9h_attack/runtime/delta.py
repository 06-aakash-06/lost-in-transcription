"""Inference-only vanilla LoRA arithmetic, preserving FP32 delta precision."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

class DeltaLinear(nn.Module):
    def __init__(self,base,a,b,scale):
        super().__init__();self.base_layer=base;self.scale=scale
        self.register_buffer('delta_a',a.to(device=base.weight.device,dtype=torch.float32))
        self.register_buffer('delta_b',b.to(device=base.weight.device,dtype=torch.float32))
    @property
    def weight(self):return self.base_layer.weight
    @property
    def bias(self):return self.base_layer.bias
    def forward(self,x):
        result=self.base_layer(x);dtype=result.dtype
        delta=F.linear(F.linear(x.to(torch.float32),self.delta_a),self.delta_b)*self.scale
        return (result+delta).to(dtype)

def inject(model,weights,scale):
    names=list(model.named_modules());consumed=set();installed=0
    for name,module in names:
        a='base_model.model.'+name+'.lora_A.weight';b='base_model.model.'+name+'.lora_B.weight'
        if a not in weights:continue
        if hasattr(module,'base_layer'):module=module.base_layer
        if not isinstance(module,nn.Linear):raise ValueError('unsupported LoRA target')
        left,right=weights[a],weights[b]
        if left.ndim!=2 or right.ndim!=2 or left.shape[1]!=module.in_features or right.shape[0]!=module.out_features or left.shape[0]!=right.shape[1]:raise ValueError('LoRA shape mismatch')
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():raise ValueError('invalid LoRA weights')
        parent,_,leaf=name.rpartition('.');owner=model.get_submodule(parent) if parent else model
        setattr(owner,leaf,DeltaLinear(module,left,right,scale));consumed.update([a,b]);installed+=1
    if consumed!=set(weights) or installed==0:raise ValueError('incomplete LoRA installation')
    model.eval();return installed

def load(model,path):
    config=json.loads((path/'lora_runtime.json').read_text())
    weights=load_file(str(path/'domain_delta.safetensors'),device='cpu')
    return inject(model,weights,config['scale'])
