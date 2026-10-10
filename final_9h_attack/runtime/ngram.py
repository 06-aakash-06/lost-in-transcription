"""Portable absolute-discount trigram candidate scoring; no free rewriting."""
import json,math
from collections import Counter,defaultdict
from .fusion import key,repetition
from .confidence import score_pair
from .audio import RATE
from .postprocess import clean

def words(text):return [w for token in text.split() if (w:=key(token))]

def train(texts):
    bi=Counter();tri=Counter();predecessors=defaultdict(set)
    for text in texts:
        seq=['<s>','<s>',*words(text),'</s>']
        for i in range(2,len(seq)):
            bi[(seq[i-1],seq[i])]+=1;tri[(seq[i-2],seq[i-1],seq[i])]+=1;predecessors[seq[i]].add(seq[i-1])
    return {'discount':.75,'bigram':{'\t'.join(k):v for k,v in bi.items()},'trigram':{'\t'.join(k):v for k,v in tri.items()},'continuation':{k:len(v) for k,v in predecessors.items()}}

class LM:
    def __init__(self,data):
        self.d=data['discount'];self.bi={tuple(k.split('\t')):v for k,v in data['bigram'].items()};self.tri={tuple(k.split('\t')):v for k,v in data['trigram'].items()};self.uni=data['continuation'];self.total=sum(self.uni.values())+1
        self.bt=Counter();self.btypes=Counter();self.tt=Counter();self.ttypes=Counter()
        for (a,b),n in self.bi.items():self.bt[a]+=n;self.btypes[a]+=1
        for (a,b,c),n in self.tri.items():self.tt[(a,b)]+=n;self.ttypes[(a,b)]+=1
    def prob(self,a,b,c):
        lower=self.uni.get(c,1)/self.total
        if self.bt[b]:lower=max(self.bi.get((b,c),0)-self.d,0)/self.bt[b]+self.d*self.btypes[b]/self.bt[b]*lower
        if self.tt[(a,b)]:lower=max(self.tri.get((a,b,c),0)-self.d,0)/self.tt[(a,b)]+self.d*self.ttypes[(a,b)]/self.tt[(a,b)]*lower
        return max(lower,1e-12)
    def score(self,text):
        seq=['<s>','<s>',*words(text),'</s>']
        return sum(math.log(self.prob(*seq[i-2:i+1])) for i in range(2,len(seq)))/(len(seq)-2)

def acoustic(values):
    available=[v for v in values if v is not None]
    return sum(available)/len(available) if available else None

def eligible(anchor,consensus_text,values,duration_s):
    score=acoustic(values)
    return duration_s<=30 and anchor!=consensus_text and score is not None and score<math.log(.8)

def generate(engine,wave,anchor):
    import torch
    assert len(wave)<=30*RATE
    features=engine.processor.feature_extractor(wave,sampling_rate=RATE,return_tensors='pt',return_attention_mask=True)
    with torch.inference_mode():
        ids=engine.model.generate(input_features=features.input_features.to(engine.device,dtype=engine.dtype),attention_mask=features.attention_mask.to(engine.device),language='indonesian',task='transcribe',num_beams=3,num_return_sequences=3,do_sample=False,return_timestamps=False,max_new_tokens=440)
    candidates=list(dict.fromkeys([anchor,*[clean(s) for s in engine.processor.batch_decode(ids,skip_special_tokens=True)]]))
    candidates=[c for c in candidates if c and .85<=len(c.split())/max(1,len(anchor.split()))<=1.15 and repetition(c)<=repetition(anchor)]
    assert candidates and candidates[0]==anchor
    return candidates,score_pair(engine,wave,candidates)

def choose(anchor,candidates,scores,lm):
    base=acoustic(scores[0])
    if base is None:return anchor
    best=anchor;best_score=base+.1*lm.score(anchor)+.02
    for candidate,values in zip(candidates[1:],scores[1:]):
        ac=acoustic(values)
        if ac is not None and (value:=ac+.1*lm.score(candidate))>best_score:best=candidate;best_score=value
    return best
