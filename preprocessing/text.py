"""Text preprocessing utilities.

Functions:
- clean_text(text): lowercase, remove punctuation, normalized whitespace
- tokenize(text): simple regex tokenizer returning list of tokens
"""
import re
from typing import List


_TOKEN_RE = re.compile(r"\b\w+\b")


def clean_text(text: str) -> str:
    if text is None:
        return ''
    text = text.strip()
    text = text.lower()
    # remove punctuation except word characters and spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())
