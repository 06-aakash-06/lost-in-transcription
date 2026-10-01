"""Deterministic, metadata-blind corrections to a Whisper surface transcript."""
import math,re,unicodedata
from dataclasses import dataclass

def key(word):
    return unicodedata.normalize('NFC',word).casefold().strip('.,!?;:…“”"()[]')

@dataclass(frozen=True)
class Block:
    start:int
    end:int
    other_start:int
    other_end:int
    left:int
    right:int

def alignment(a,b):
    """Minimum word edit alignment, retaining original word indices."""
    x,y=list(map(key,a)),list(map(key,b));n,m=len(x),len(y)
    d=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):d[i][0]=i
    for j in range(m+1):d[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):d[i][j]=min(d[i-1][j-1]+(x[i-1]!=y[j-1]),d[i-1][j]+1,d[i][j-1]+1)
    out=[];i,j=n,m
    while i or j:
        if i and j and d[i][j]==d[i-1][j-1]+(x[i-1]!=y[j-1]):
            out.append((i-1,j-1,x[i-1]==y[j-1]));i-=1;j-=1
        elif i and d[i][j]==d[i-1][j]+1:out.append((i-1,None,False));i-=1
        else:out.append((None,j-1,False));j-=1
    return out[::-1]

def blocks(a,b):
    pairs=alignment(a,b);out=[];ai=bi=0;k=0
    while k<len(pairs):
        if pairs[k][2]:ai+=1;bi+=1;k+=1;continue
        begin=k;s,t=ai,bi
        while k<len(pairs) and not pairs[k][2]:
            i,j,_=pairs[k];ai+=i is not None;bi+=j is not None;k+=1
        left=0;z=begin-1
        while z>=0 and pairs[z][2]:left+=1;z-=1
        right=0;z=k
        while z<len(pairs) and pairs[z][2]:right+=1;z+=1
        out.append(Block(s,ai,t,bi,left,right))
    return out

def repetition(text,min_repeats=5):
    """Return fraction occupied by a clearly repeated 1--6 word phrase."""
    w=list(map(key,text.split()));best=0
    for width in range(1,7):
        i=0
        while i+width*min_repeats<=len(w):
            phrase=w[i:i+width];j=i+width
            while w[j:j+width]==phrase:j+=width
            if (j-i)//width>=min_repeats:best=max(best,j-i)
            i+=1
    return best/max(1,len(w))

def collapse_single_runs(text):
    w=text.split();out=[];i=0
    while i<len(w):
        j=i+1
        while j<len(w) and key(w[j])==key(w[i]):j+=1
        out.extend(w[i:i+(2 if j-i>=5 else j-i)]);i=j
    return ' '.join(out)

def average_log(values,start,end):
    if not values or start==end:return None
    selected=values[start:end]
    if len(selected)!=end-start or any(v is None for v in selected):return None
    return sum(selected)/len(selected)

def fuse(anchor,meralion,config,confidence=None,third=None):
    a,b=anchor.split(),meralion.split()
    if config.get('collapse_anchor',False):
        collapsed=collapse_single_runs(anchor)
        if collapsed!=anchor:
            # Confidence indices remain tied to the raw anchor; no further
            # edits after a proven catastrophic-run correction.
            return collapsed
    ar,br=repetition(anchor),repetition(meralion)
    if config.get('repeat_swap',False) and ar>=.35 and br==0 and len(b)>=4 and len(b)<.8*len(a):return meralion.strip()
    if br>=.20:return anchor.strip()
    if not a or not b:return anchor.strip()
    c=(confidence or {}).get('anchor');other=(confidence or {}).get('meralion')
    support={}
    if third is not None:
        tw=third.split()
        for block in blocks(a,tw):support[(block.start,block.end)]=[key(w) for w in tw[block.other_start:block.other_end]]
    chosen=[]
    for block in blocks(a,b):
        na,nb=block.end-block.start,block.other_end-block.other_start
        operation='substitution' if na and nb else 'insertion' if nb else 'deletion'
        if operation not in config.get('operations',['substitution']):continue
        if max(na,nb)>config.get('max_span',2):continue
        if min(block.left,block.right)<config.get('context',1):continue
        candidate=b[block.other_start:block.other_end]
        is_supported=support.get((block.start,block.end))==[key(w) for w in candidate] if third is not None else False
        if config.get('mode')=='support':
            approve=is_supported
        elif config.get('mode') in ('confidence','hybrid'):
            lp=average_log(c,block.start,block.end)
            mp=average_log(other,block.other_start,block.other_end)
            if na==0:
                nearby=[v for v in (c or [])[max(0,block.start-1):block.start+1] if v is not None]
                lp=min(nearby) if nearby else None
            approve=lp is not None and lp<math.log(config.get('threshold',.6))
            if config.get('margin') is not None:approve=approve and mp is not None and mp-lp>=config['margin']
            if operation=='deletion':approve=approve and config.get('allow_deletion',False)
            if config.get('require_support'):approve=approve and is_supported
        else:
            # A sequence-only rule is deliberately narrow: remove obvious
            # adjacent duplicates when the other hypothesis has one copy.
            approve=operation=='deletion' and na==1 and ((block.start and key(a[block.start])==key(a[block.start-1])) or (block.end<len(a) and key(a[block.start])==key(a[block.end])))
        if config.get('mode')=='hybrid':
            approve=(approve and max(na,nb)<=config.get('confidence_span',2) and operation=='substitution') or is_supported
        if approve:chosen.append((block.start,block.end,candidate))
    out=[];position=0
    for start,end,candidate in chosen:out.extend(a[position:start]);out.extend(candidate);position=end
    out.extend(a[position:]);return ' '.join(out)
