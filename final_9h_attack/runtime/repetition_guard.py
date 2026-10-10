"""Retain two cycles of a clearly dominant, catastrophic phrase loop."""
from .fusion import key,collapse_single_runs

def guard(text):
    text=collapse_single_runs(text);original=text.split();normal=list(map(key,original));out=[];i=0
    while i<len(original):
        best=None
        for width in range(2,7):
            if i+5*width>len(original):continue
            phrase=normal[i:i+width];end=i+width
            while end+width<=len(original) and normal[end:end+width]==phrase:end+=width
            span=end-i
            if span//width>=5 and span>=20 and span>=.35*len(original):
                if best is None or span>best[0]:best=(span,width,end)
        if best is None:out.append(original[i]);i+=1
        else:
            _,width,end=best;out.extend(original[i:i+2*width]);i=end
    return ' '.join(out)
