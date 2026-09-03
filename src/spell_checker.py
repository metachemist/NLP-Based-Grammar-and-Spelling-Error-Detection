"""Layer 2 -- non-word spelling errors.

Method: dictionary lookup + edit distance (the noisy-channel model).
  1. If a token is not a known English word...
  2. ...generate every dictionary word within edit distance <= 2
     (insertions, deletions, substitutions, transpositions), then
  3. rank those candidates by how frequent they are in English.

`pyspellchecker` implements exactly this (it is Peter Norvig's algorithm
with a 100k-word frequency list bundled in).
"""

from __future__ import annotations

import functools

from .issue import Issue


# pyspellchecker ships a US-English frequency list. Extend it with common
# British spellings and a few tech words so they are not flagged as errors.
_EXTRA_WORDS = (
    "programme programmes colour colours favour favours behaviour behaviours "
    "neighbour neighbours labour honour humour flavour organise organised "
    "organisation recognise recognised analyse analysed centre centres theatre "
    "metre litre fibre licence defence offence practise travelling travelled "
    "labelled modelling cancelled catalogue dialogue analogue grey "
    "dataset datasets tokeniser tokenisation preprocessing wifi email website "
    "spacy pipeline pipelines"
).split()


@functools.lru_cache(maxsize=1)
def _get_spell():
    from spellchecker import SpellChecker

    spell = SpellChecker(distance=2)  # edit distance 2 = catches most real typos
    spell.word_frequency.load_words(_EXTRA_WORDS)
    return spell


def _rank(spell, candidates: set[str]) -> list[str]:
    """Best-first, by corpus frequency (the P(word) term of the model)."""
    return sorted(candidates, key=lambda w: spell.word_frequency[w], reverse=True)


# Fragments spaCy splits off contractions ("don't" -> "do" + "n't",
# "dont" -> "do" + "nt"). Not real words; not our job to flag.
_CLITICS = {"n't", "nt", "'s", "s", "'re", "re", "'ve", "ve",
            "'ll", "ll", "'d", "d", "'m", "m"}


def _should_check(token) -> bool:
    if not token.is_alpha:
        return False  # numbers, punctuation, hyphenated, contractions
    if token.text.lower() in _CLITICS:
        return False
    if len(token.text) == 1:
        return False
    if token.like_url or token.like_email:
        return False
    if token.text.isupper():
        return False  # acronyms: NASA, HTTP
    if token.pos_ == "PROPN":
        return False  # names, places -- not in a general dictionary
    return True


def check_spelling(doc) -> list[Issue]:
    spell = _get_spell()
    issues: list[Issue] = []

    for token in doc:
        if not _should_check(token):
            continue

        word = token.text
        lower = word.lower()

        if spell.known([lower]):
            continue  # it's a real word

        candidates = spell.candidates(lower) or set()
        ranked = _rank(spell, candidates)

        # Preserve the original capitalisation in the suggestions.
        if word[0].isupper():
            ranked = [c.capitalize() for c in ranked]

        issues.append(
            Issue(
                start=token.idx,
                end=token.idx + len(word),
                text=word,
                category="spelling",
                rule_id="NON_WORD",
                message=(
                    f'"{word}" is not in the dictionary.'
                    + (f' Did you mean "{ranked[0]}"?' if ranked else "")
                ),
                suggestions=ranked[:5],
                source="spellchecker",
            )
        )

    return issues
