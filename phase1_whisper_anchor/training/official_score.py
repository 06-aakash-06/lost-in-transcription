# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jiwer",
#     "pandas",
#     "typer",
# ]
# ///
import re
from pathlib import Path



def _lowercase_sentence_initial(match):
    """Lowercase a sentence-initial letter.

    Keeps the letter uppercase when the next letter is also uppercase, which marks
    the start of an acronym or initialism.

    Input: a `re.Match` from the SENTENCE_INITIAL pattern with three groups:
        1. sentence delimiter (start-of-string or `.!?—` plus optional space)
        2. first letter of the sentence
        3. next letter (may be empty)
    Output: the joined string with group 2 possibly lowercased.
    """
    delimiter, first, second = match.group(1), match.group(2), match.group(3)
    if first.isupper() and not (second and second.isupper()):
        first = first.lower()
    return delimiter + first + second


def normalize_text(text):
    """Normalize a transcript to a canonical form for scoring.

    Removes bracketed annotations (`[...]`), unintelligible markers (`(?)`),
    parenthetical wrappers, and stray punctuation (`¿¡";:!?`). Lowercases
    sentence-initial letters. Rewrites em dashes as commas, then drops commas
    and periods (but preserves `...`). Collapses runs of whitespace.

    Input: a raw transcript string.
    Output: the cleaned transcript string.
    """
    BRACKETED = re.compile(r"\[[^\]]+\]")
    UNINTELLIGIBLE_PAREN = re.compile(r"\(\?+\)")
    WORD_PAREN = re.compile(r"\(([^()]*)\)")
    PUNCTUATION_OTHER = re.compile('[¿¡";:]+')
    COMMA = re.compile(",+")
    # [^\W\d_] matches any Unicode letter (word char minus digits/underscore),
    # including sentence-initial accented capitals (e.g. Spanish Á/É/Í/Ó/Ú/Ñ)
    SENTENCE_INITIAL = re.compile(r"(^\s*|[.!?—]\s*)([^\W\d_])([^\W\d_]?)")
    SENTENCE_END = re.compile("[!?]+")
    MULTISPACE = re.compile("  +")

    text = text.replace("~", "")
    text = re.sub(BRACKETED, " ", text)
    text = re.sub(UNINTELLIGIBLE_PAREN, " ", text)
    text = re.sub(WORD_PAREN, r"\1", text)
    text = text.replace("#x27;", "'")
    text = re.sub(PUNCTUATION_OTHER, " ", text)
    text = re.sub(SENTENCE_INITIAL, _lowercase_sentence_initial, text)
    # self-interruption em dash becomes a comma + space, same as any other comma, so it
    # collapses away to a single space by the time normalization finishes
    text = text.replace("—", ", ")
    text = re.sub(COMMA, " ", text)
    text = re.sub(SENTENCE_END, " ", text)
    text = text.replace("...", "!ELLIPSIS!").replace(".", " ").replace("!ELLIPSIS!", "...")
    while " ... " in text:
        text = text.replace(" ... ", " ")
    text = re.sub(MULTISPACE, " ", text)
    return text
