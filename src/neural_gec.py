"""Layer 4 -- Transformer-based grammatical error correction (optional).

Idea: instead of *detecting* errors, use a sequence-to-sequence model that
was fine-tuned on millions of (ungrammatical -> grammatical) sentence pairs
to simply *rewrite* the sentence correctly. Then diff the rewrite against
the original: every span the model changed is a detected error.

Model: `vennify/t5-base-grammar-correction` (a T5-base fine-tune, ~850 MB).
It downloads on first use and is cached by Hugging Face afterwards.

This module is import-safe even when `transformers`/`torch` are missing, so
the rest of the app runs without them.
"""

from __future__ import annotations

import difflib
import functools
import os
import re

from .issue import Issue

# Hugging Face repo id by default; point NEURAL_GEC_MODEL at a local directory
# (containing config.json, the weights, and the tokenizer files) to run offline.
_MODEL = os.environ.get("NEURAL_GEC_MODEL", "vennify/t5-base-grammar-correction")


def is_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


@functools.lru_cache(maxsize=1)
def _get_model():
    """Load tokenizer + seq2seq model once.

    We drive the model directly rather than via ``transformers.pipeline``:
    the ``text2text-generation`` pipeline task was removed in transformers 5,
    and ``AutoModelForSeq2SeqLM.generate`` is stable across 4.x and 5.x.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL)
    model.eval()
    return tokenizer, model


def _correct_sentence(sentence: str) -> str:
    import torch

    tokenizer, model = _get_model()
    inputs = tokenizer(
        "grammar: " + sentence, return_tensors="pt",
        truncation=True, max_length=256,
    )
    with torch.no_grad():
        out = model.generate(
            **inputs, num_beams=4, do_sample=False, max_length=256,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _diff_to_issues(original: str, corrected: str, offset: int) -> list[Issue]:
    """Word-level diff -> Issues, with char offsets back into the full text."""
    src_tokens = list(re.finditer(r"\S+", original))
    src_words = [m.group(0) for m in src_tokens]
    tgt_words = corrected.split()

    if [w.lower() for w in src_words] == [w.lower() for w in tgt_words]:
        return []

    issues: list[Issue] = []
    sm = difflib.SequenceMatcher(
        a=[w.lower() for w in src_words], b=[w.lower() for w in tgt_words]
    )
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        replacement = " ".join(tgt_words[j1:j2])

        if tag in ("replace", "delete"):
            start = offset + src_tokens[i1].start()
            end = offset + src_tokens[i2 - 1].end()
            bad = original[src_tokens[i1].start():src_tokens[i2 - 1].end()]
            msg = (f'Suggested change: "{bad}" → "{replacement}".'
                   if replacement else f'Consider removing "{bad}".')
        else:  # insert
            anchor = src_tokens[i1 - 1].end() if i1 > 0 else 0
            start = end = offset + anchor
            bad = ""
            msg = f'Consider inserting "{replacement}".'

        issues.append(Issue(
            start=start, end=end, text=bad, category="grammar",
            rule_id="NEURAL_GEC", message=msg,
            suggestions=[replacement] if replacement else [],
            source="neural",
        ))
    return issues


def neural_issues(doc) -> list[Issue]:
    """`doc` is a spaCy Doc (we use it for sentence offsets)."""
    issues: list[Issue] = []
    for sent in doc.sents:
        text = sent.text.strip()
        if len(text) < 2:
            continue
        corrected = _correct_sentence(text)
        issues.extend(_diff_to_issues(sent.text, corrected, offset=sent.start_char))
    return issues
