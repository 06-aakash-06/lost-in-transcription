"""Small edit gate with inference-only features and plain JSON weights."""
import math
from .fusion import blocks, key, repetition, collapse_single_runs, average_log, alignment

FEATURES = ['word_logp','word_missing','acoustic_logp','surrounding_logp',
            'support','substitution','insertion','deletion','anchor_span',
            'other_span','span_edit_distance','anchor_repetition','other_repetition',
            'agreement_count']

def proposals(anchor, other, confidence, third):
    a,b=anchor.split(),other.split()
    if not a or not b or repetition(other)>=.20 or collapse_single_runs(anchor)!=anchor:
        return []
    c=(confidence or {}).get('anchor') or []
    support={}
    for block in blocks(a,third.split()):
        support[block.start,block.end]=[key(w) for w in third.split()[block.other_start:block.other_end]]
    available=[v for v in c if v is not None]
    acoustic=sum(available)/len(available) if available else 0.
    out=[]
    for block in blocks(a,b):
        na,nb=block.end-block.start,block.other_end-block.other_start
        if max(na,nb)>3 or min(block.left,block.right)<1:continue
        candidate=b[block.other_start:block.other_end]
        supported=support.get((block.start,block.end))==[key(w) for w in candidate]
        lp=average_log(c,block.start,block.end)
        nearby=[v for v in c[max(0,block.start-1):min(len(c),block.end+1)] if v is not None]
        if na==0:lp=min(nearby) if nearby else None
        operation='substitution' if na and nb else 'insertion' if nb else 'deletion'
        # Count minimum word edit operations without reference access.
        distance=sum(not equal for _,_,equal in alignment(a[block.start:block.end],candidate))
        features=[lp if lp is not None else 0.,float(lp is None),acoustic,
                  sum(nearby)/len(nearby) if nearby else 0.,float(supported),
                  float(operation=='substitution'),float(operation=='insertion'),float(operation=='deletion'),
                  na,nb,distance,repetition(anchor),repetition(other),1+int(supported)]
        baseline=supported or (operation=='substitution' and max(na,nb)<=2 and lp is not None and lp<math.log(.65))
        out.append((block.start,block.end,candidate,features,baseline))
    return out

def apply_edits(anchor, edits):
    a=anchor.split();out=[];position=0
    for start,end,candidate in edits:
        out.extend(a[position:start]);out.extend(candidate);position=end
    return ' '.join(out+a[position:])

def gated_fuse(anchor, other, confidence, third, gate):
    collapsed=collapse_single_runs(anchor)
    if collapsed!=anchor:return collapsed
    selected=[]
    for start,end,candidate,features,baseline in proposals(anchor,other,confidence,third):
        if gate['scope']=='baseline' and not baseline:continue
        score=gate['intercept']+sum(w*(x-m)/s for w,x,m,s in zip(gate['coef'],features,gate['mean'],gate['scale']))
        if score>=0:selected.append((start,end,candidate))
    return apply_edits(anchor,selected)
