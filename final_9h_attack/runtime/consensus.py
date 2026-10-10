"""Conservative word voting with unique optimal edit alignment."""
from .fusion import key,collapse_single_runs,repetition

def unique_map(a,b):
    x,y=list(map(key,a)),list(map(key,b));n,m=len(x),len(y)
    forward=[[0]*(m+1) for _ in range(n+1)];reverse=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):forward[i][0]=i;reverse[i][m]=n-i
    for j in range(m+1):forward[0][j]=j;reverse[n][j]=m-j
    for i in range(1,n+1):
        for j in range(1,m+1):forward[i][j]=min(forward[i-1][j-1]+(x[i-1]!=y[j-1]),forward[i-1][j]+1,forward[i][j-1]+1)
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):reverse[i][j]=min(reverse[i+1][j+1]+(x[i]!=y[j]),reverse[i+1][j]+1,reverse[i][j+1]+1)
    optimum=forward[n][m];mapping={}
    for i in range(n):
        if any(forward[i][j]+1+reverse[i+1][j]==optimum for j in range(m+1)):continue
        js=[j for j in range(m) if forward[i][j]+(x[i]!=y[j])+reverse[i+1][j+1]==optimum]
        if len(js)==1:mapping[i]=js[0]
    return mapping

def proposals(anchor,first,second,context=0):
    a,b,c=anchor.split(),first.split(),second.split();mb,mc=unique_map(a,b),unique_map(a,c);out=[]
    for i in range(len(a)):
        if i not in mb or i not in mc:continue
        candidate=b[mb[i]]
        if candidate!=c[mc[i]] or key(candidate)==key(a[i]):continue
        if context and (i==0 or i==len(a)-1):continue
        if context and any(j not in mb or j not in mc or key(a[j])!=key(b[mb[j]]) or key(a[j])!=key(c[mc[j]]) for j in [i-1,i+1]):continue
        out.append((i,candidate))
    return out

def consensus(anchor,first,second,context=0,confidence=None,threshold=None,gate=None):
    raw=anchor;anchor=collapse_single_runs(anchor)
    # Scoring positions belong to the original words. Stop after loop cleanup.
    if anchor!=raw:return anchor
    a=anchor.split();out=a.copy();values=(confidence or {}).get('anchor') or []
    for i,candidate in proposals(anchor,first,second,context):
        if threshold is not None:
            import math
            if i>=len(values) or values[i] is None or values[i]>=math.log(threshold):continue
        if gate is not None:
            features=gate_features(anchor,first,second,i,values)
            z=gate['intercept']+sum(w*(v-m)/s for w,v,m,s in zip(gate['coef'],features,gate['mean'],gate['scale']))
            if z<0:continue
        out[i]=candidate
    return ' '.join(out)

FEATURES=['word_logp','word_missing','surrounding_logp','acoustic_logp','relative_turbo_length','relative_mer_length','local_disagreement_density','anchor_repetition','turbo_repetition','mer_repetition','at_boundary','agreement_count']

def gate_features(anchor,first,second,i,values):
    a,b,c=anchor.split(),first.split(),second.split();lp=values[i] if i<len(values) else None
    nearby=[v for v in values[max(0,i-1):i+2] if v is not None];available=[v for v in values if v is not None]
    edits={j for j,_ in proposals(anchor,first,second,0)}
    density=sum(j in edits for j in range(max(0,i-2),min(len(a),i+3)))/max(1,min(len(a),i+3)-max(0,i-2))
    return [lp if lp is not None else 0.,float(lp is None),sum(nearby)/len(nearby) if nearby else 0.,sum(available)/len(available) if available else 0.,len(b)/max(1,len(a)),len(c)/max(1,len(a)),density,repetition(anchor),repetition(first),repetition(second),float(i==0 or i==len(a)-1),2.]
