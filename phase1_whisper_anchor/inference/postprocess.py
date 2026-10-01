"""Minimal whitespace cleanup and conservative overlap merge."""
from __future__ import annotations
import re

def clean(text: str) -> str:
    if not isinstance(text, str): raise TypeError("model output must be text")
    return re.sub(r"\s+", " ", text).strip()

def merge_overlap(parts: list[str], max_words: int = 12) -> str:
    if not parts: return ""
    words = clean(parts[0]).split()
    for part in parts[1:]:
        following = clean(part).split()
        overlap = 0
        for n in range(1, min(max_words, len(words), len(following))+1):
            if [x.casefold() for x in words[-n:]] == [x.casefold() for x in following[:n]]: overlap=n
        words.extend(following[overlap:])
    return " ".join(words)

def catastrophic_repeat(text: str, min_repeats: int = 5) -> bool:
    words = clean(text).split()
    return any(len(set(w.casefold() for w in words[i:i+min_repeats])) == 1 for i in range(max(0,len(words)-min_repeats+1)))

def collapse_catastrophic(text: str, min_repeats: int = 5) -> str:
    """Optional diagnostic only; keep first two tokens of a 5+ identical run."""
    words=clean(text).split();out=[];i=0
    while i<len(words):
        j=i+1
        while j<len(words) and words[j].casefold()==words[i].casefold(): j+=1
        out.extend(words[i:i+(2 if j-i>=min_repeats else j-i)])
        i=j
    return " ".join(out)
