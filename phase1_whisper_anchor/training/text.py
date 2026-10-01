"""Conservative training target conversion, never used to score predictions."""
from __future__ import annotations
import re
import unicodedata

MAP = {"“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "–": "-", "—": " "}

def to_reference_style(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("training transcript must be a string")
    text = unicodedata.normalize("NFC", text)
    for before, after in MAP.items():
        text = text.replace(before, after)
    # Existing Jember preparation already applied its audited spelling rules.
    # Keep accents, casing, fillers, repeated words, and reduplication intact.
    return re.sub(r"\s+", " ", text).strip()
