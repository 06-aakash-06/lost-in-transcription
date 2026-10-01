"""Official normalizer plus corpus-level Levenshtein operation counts."""
from __future__ import annotations
try:
    from .official_score import normalize_text
except ImportError:
    from official_score import normalize_text

def counts(reference: str, prediction: str) -> tuple[int, int, int, int]:
    ref, hyp = normalize_text(reference).split(), normalize_text(prediction).split()
    dp = [[(0, 0, 0, 0)] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1): dp[i][0] = (i, 0, i, 0)
    for j in range(1, len(hyp) + 1): dp[0][j] = (j, 0, 0, j)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]: dp[i][j] = dp[i-1][j-1]
            else:
                a = dp[i-1][j-1]; b = dp[i-1][j]; c = dp[i][j-1]
                dp[i][j] = min((a[0]+1,a[1]+1,a[2],a[3]), (b[0]+1,b[1],b[2]+1,b[3]), (c[0]+1,c[1],c[2],c[3]+1))
    _, s, d, ins = dp[-1][-1]
    return len(ref), s, d, ins

def corpus(references: list[str], predictions: list[str]) -> dict[str, float | int]:
    if len(references) != len(predictions): raise ValueError("prediction count mismatch")
    n=s=d=i=0
    for ref, hyp in zip(references, predictions):
        a,b,c,e = counts(ref,hyp); n+=a;s+=b;d+=c;i+=e
    return {"wer": (s+d+i)/n if n else 0.0, "words": n, "substitutions": s, "deletions": d, "insertions": i}
