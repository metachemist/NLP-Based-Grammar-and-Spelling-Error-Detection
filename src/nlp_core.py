"""Loads spaCy once and hands the same pipeline to every layer.

spaCy's `en_core_web_sm` gives us, in one pass over the text:
  - tokenization      (token.text, token.idx = char offset in original text)
  - sentence segmentation   (doc.sents)
  - part-of-speech tags     (token.tag_  -> fine-grained: VBZ, NNS, ...)
  - a dependency parse      (token.dep_, token.head -> who is subject of what)

Loading the model takes ~1s, so we cache it.
"""

from __future__ import annotations

import functools
import sys

_MODEL = "en_core_web_sm"


@functools.lru_cache(maxsize=1)
def get_nlp():
    try:
        import spacy
    except ImportError:
        sys.exit("spaCy not installed. Run: pip install -r requirements.txt")

    try:
        return spacy.load(_MODEL)
    except OSError:
        sys.exit(
            f"spaCy model '{_MODEL}' not found.\n"
            f"Run: python -m spacy download {_MODEL}"
        )


def analyze_text(text: str):
    """Return a spaCy Doc: an iterable of tokens, each with offsets + tags."""
    return get_nlp()(text)
