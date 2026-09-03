"""Layer 3 -- rule-based grammar checking.

Each function is one explicit rule that reads spaCy's part-of-speech tags
(token.tag_ : VBZ, VBP, NNS, ...) and dependency parse (token.dep_,
token.head) and returns Issues. These rules have very high precision
(they almost never cry wolf) but low recall (they only know the mistakes
we thought to write down). Layer 4 (the neural model) covers the rest.

Fine-grained POS tags used below:
  VBZ = verb, 3rd person singular present  ("goes", "is")
  VBP = verb, non-3rd person present       ("go", "are")
  NN  = singular noun     NNS = plural noun
"""

from __future__ import annotations

import re

from .issue import Issue

_VOWEL_LETTERS = set("aeiou")

# start with a vowel letter, but a consonant *sound* -> take "a"
_A_BEFORE = {
    "university", "universe", "unicorn", "unit", "unique", "uniform",
    "user", "useful", "european", "one", "once", "eulogy", "ewe",
}
# start with a consonant letter, but a vowel *sound* -> take "an"
_AN_BEFORE = {"hour", "honest", "honor", "honour", "heir", "honestly", "hourly"}

_SINGULAR_PRONOUNS = {
    "he", "she", "it", "this", "that",
    "everyone", "someone", "anyone", "nobody", "everybody", "somebody",
}
_PLURAL_PRONOUNS = {"they", "we", "you", "these", "those"}

_SENTENCE_ENDERS = {".", "!", "?", "…", ":", ";"}
# closing marks that can legitimately follow the real end of a sentence
_CLOSERS = {'"', "'", ")", "]", "}", "”", "’", "»", "``", "''"}

_BE_SINGULAR = {"is", "was"}
_BE_PLURAL = {"are", "were"}
_BE_FIX = {"is": "are", "are": "is", "was": "were", "were": "was"}


# --------------------------------------------------------------------------- #
# individual rules
# --------------------------------------------------------------------------- #
def _rule_sentence_capitalization(doc) -> list[Issue]:
    out = []
    for sent in doc.sents:
        for token in sent:
            if token.is_space or token.is_punct:
                continue
            if token.is_alpha and token.text[0].islower() and token.pos_ != "X":
                out.append(Issue(
                    start=token.idx, end=token.idx + len(token.text),
                    text=token.text, category="grammar",
                    rule_id="SENTENCE_CAPITALIZATION",
                    message=f'Sentences should start with a capital letter: "{token.text}".',
                    suggestions=[token.text.capitalize()], source="rules",
                ))
            break  # only inspect the first real token of the sentence
    return out


def _rule_standalone_i(doc) -> list[Issue]:
    out = []
    for token in doc:
        if token.text == "i" and token.is_alpha:
            out.append(Issue(
                start=token.idx, end=token.idx + 1, text="i", category="grammar",
                rule_id="LOWERCASE_I",
                message='The pronoun "I" is always capitalised.',
                suggestions=["I"], source="rules",
            ))
    return out


def _wants_an(word: str) -> bool:
    w = word.lower()
    if w in _A_BEFORE:
        return False
    if w in _AN_BEFORE:
        return True
    return bool(w) and w[0] in _VOWEL_LETTERS


def _rule_a_vs_an(doc) -> list[Issue]:
    out = []
    for i, token in enumerate(doc):
        if token.lower_ not in ("a", "an") or i + 1 >= len(doc):
            continue
        nxt = doc[i + 1]
        if not nxt.is_alpha:
            continue
        cap = token.text[0].isupper()
        if token.lower_ == "a" and _wants_an(nxt.text):
            fix = "An" if cap else "an"
        elif token.lower_ == "an" and not _wants_an(nxt.text):
            fix = "A" if cap else "a"
        else:
            continue
        out.append(Issue(
            start=token.idx, end=token.idx + len(token.text), text=token.text,
            category="grammar", rule_id="A_VS_AN",
            message=f'Use "{fix}" before "{nxt.text}".',
            suggestions=[fix], source="rules",
        ))
    return out


_IRREGULAR_3SG = {
    "be": "is", "have": "has", "do": "does", "go": "goes", "say": "says",
}


def _third_person_singular(lemma: str) -> str:
    if lemma in _IRREGULAR_3SG:
        return _IRREGULAR_3SG[lemma]
    if re.search(r"(s|x|z|o|ch|sh)$", lemma):
        return lemma + "es"
    if re.search(r"[^aeiou]y$", lemma):
        return lemma[:-1] + "ies"
    return lemma + "s"


def _subject_number(token):
    """Return 'sing', 'plur', or None for a subject token."""
    if token.lower_ == "i":
        return None  # "I go" is correct; don't touch it
    if token.lower_ in _SINGULAR_PRONOUNS or token.tag_ in ("NN", "NNP"):
        return "sing"
    if token.lower_ in _PLURAL_PRONOUNS or token.tag_ in ("NNS", "NNPS"):
        return "plur"
    return None


def _agreement_bearer(subject):
    """The token that must agree with the subject.

    Usually the head verb itself, but for "friends *is* coming" / "children
    *was* playing" the head is a participle and agreement lives on the
    auxiliary. Returns (token, is_auxiliary) or (None, None).
    """
    verb = subject.head
    if verb.pos_ not in ("VERB", "AUX"):
        return None, None
    if verb.tag_ in ("VBZ", "VBP"):
        return verb, False
    for child in verb.children:
        if child.dep_ in ("aux", "auxpass") and child.lemma_ in ("be", "have", "do"):
            return child, True
    return None, None


def _rule_subject_verb_agreement(doc) -> list[Issue]:
    out = []
    for token in doc:
        if token.dep_ not in ("nsubj", "nsubjpass"):
            continue
        number = _subject_number(token)
        if number is None:
            continue
        bearer, is_aux = _agreement_bearer(token)
        if bearer is None:
            continue

        low = bearer.text.lower()
        fix = None

        if is_aux and bearer.lemma_ == "be":
            if low in _BE_SINGULAR and number == "plur":
                fix = _BE_FIX[low]
            elif low in _BE_PLURAL and number == "sing":
                fix = _BE_FIX[low]
        elif is_aux and bearer.lemma_ == "have":
            if low == "has" and number == "plur":
                fix = "have"
            elif low == "have" and number == "sing":
                fix = "has"
        elif is_aux and bearer.lemma_ == "do":
            if low == "does" and number == "plur":
                fix = "do"
            elif low == "do" and number == "sing":
                fix = "does"
        elif not is_aux:
            if bearer.tag_ == "VBP" and number == "sing":
                fix = _third_person_singular(bearer.lemma_)
            elif bearer.tag_ == "VBZ" and number == "plur":
                fix = bearer.lemma_

        if fix is None:
            continue

        kind = "singular" if number == "sing" else "plural"
        out.append(Issue(
            start=bearer.idx, end=bearer.idx + len(bearer.text), text=bearer.text,
            category="grammar", rule_id="SUBJECT_VERB_AGREEMENT",
            message=(f'{kind.capitalize()} subject "{token.text}" needs '
                     f'"{fix}", not "{bearer.text}".'),
            suggestions=[fix], source="rules",
        ))
    return out


_NON_BASE_AFTER_DO = {"VBD", "VBN", "VBZ", "VBG"}


def _rule_do_support_base_form(doc) -> list[Issue]:
    """After "do/does/did" (incl. "don't/doesn't/didn't"), the main verb
    must be the bare infinitive: "I didn't *know*", not "I didn't *knew*".

    We look for a verb whose dependency children include an ``aux`` with
    lemma "do"; if that verb is tagged as anything other than a base form
    (VBD/VBN/VBZ/VBG), we flag it and suggest the lemma.
    """
    out = []
    for token in doc:
        if token.pos_ not in ("VERB", "AUX"):
            continue
        if token.tag_ not in _NON_BASE_AFTER_DO:
            continue
        do_aux = next(
            (c for c in token.children
             if c.dep_ == "aux" and c.lemma_ == "do"),
            None,
        )
        if do_aux is None:
            continue
        base = token.lemma_
        if not base or base == token.lower_:
            continue
        fix = base.capitalize() if token.text[:1].isupper() else base
        out.append(Issue(
            start=token.idx, end=token.idx + len(token.text), text=token.text,
            category="grammar", rule_id="DO_SUPPORT_BASE_FORM",
            message=(f'After "{do_aux.text}" use the base form "{fix}", '
                     f'not "{token.text}".'),
            suggestions=[fix], source="rules",
        ))
    return out


def _rule_repeated_word(doc) -> list[Issue]:
    out = []
    for i in range(len(doc) - 1):
        a, b = doc[i], doc[i + 1]
        if a.is_alpha and b.is_alpha and a.lower_ == b.lower_:
            out.append(Issue(
                start=a.idx, end=b.idx + len(b.text),
                text=f"{a.text} {b.text}", category="grammar",
                rule_id="REPEATED_WORD",
                message=f'"{a.text}" is repeated.',
                suggestions=[a.text], source="rules",
            ))
    return out


def _rule_space_before_punctuation(doc) -> list[Issue]:
    out = []
    for m in re.finditer(r"\s+([,.!?;:])", doc.text):
        out.append(Issue(
            start=m.start(), end=m.end(), text=m.group(0),
            category="grammar", rule_id="SPACE_BEFORE_PUNCT",
            message=f'Remove the space before "{m.group(1)}".',
            suggestions=[m.group(1)], source="rules",
        ))
    return out


def _rule_missing_sentence_punctuation(doc) -> list[Issue]:
    out = []
    for sent in doc.sents:
        real = [t for t in sent if not t.is_space and t.text not in _CLOSERS]
        if len(real) < 3:
            continue
        if not any(t.tag_.startswith("VB") for t in real):
            continue  # not a full clause -- probably a heading or fragment
        last = real[-1]
        if last.text in _SENTENCE_ENDERS:
            continue
        out.append(Issue(
            start=last.idx + len(last.text), end=last.idx + len(last.text),
            text="", category="grammar", rule_id="MISSING_SENTENCE_PUNCT",
            message="This sentence has no end punctuation.",
            suggestions=["."], source="rules",
        ))
    return out


_RULES = (
    _rule_sentence_capitalization,
    _rule_standalone_i,
    _rule_a_vs_an,
    _rule_subject_verb_agreement,
    _rule_do_support_base_form,
    _rule_repeated_word,
    _rule_space_before_punctuation,
    _rule_missing_sentence_punctuation,
)


def check_grammar(doc) -> list[Issue]:
    issues: list[Issue] = []
    for rule in _RULES:
        issues.extend(rule(doc))
    return issues
