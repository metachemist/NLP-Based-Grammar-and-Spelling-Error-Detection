"""Orchestrates the layers into one deduplicated, sorted list of Issues.

    raw text
       |
   spaCy (tokenize + POS + parse)      <- nlp_core
       |
   +---+-----------------+-------------------+
   |                     |                   |
 spelling            grammar rules      neural GEC (optional)
 (Layer 2)            (Layer 3)          (Layer 4)
   |                     |                   |
   +---------------------+-------------------+
                         |
                   dedupe overlaps
                         |
                 sorted list of Issues
"""

from __future__ import annotations

from . import neural_gec
from .grammar_rules import check_grammar
from .issue import Issue
from .nlp_core import analyze_text
from .spell_checker import check_spelling

# Lower number = higher priority when two Issues cover the same span.
# A specific spelling/rule finding beats a generic "the model changed this".
_SOURCE_PRIORITY = {"spellchecker": 0, "rules": 1, "neural": 2}


def _dedupe(issues: list[Issue]) -> list[Issue]:
    ordered = sorted(
        issues, key=lambda x: (_SOURCE_PRIORITY.get(x.source, 9), x.start)
    )
    kept: list[Issue] = []
    for issue in ordered:
        if issue.start != issue.end and any(issue.overlaps(k) for k in kept):
            continue
        kept.append(issue)
    return sorted(kept, key=lambda x: (x.start, x.end))


def analyze(text: str, use_neural: bool = False) -> list[Issue]:
    doc = analyze_text(text)
    issues: list[Issue] = []
    issues += check_spelling(doc)
    issues += check_grammar(doc)

    if use_neural and neural_gec.is_available():
        try:
            issues += neural_gec.neural_issues(doc)
        except Exception as exc:  # never let the demo crash on the heavy layer
            print(f"[neural_gec] skipped: {exc}")

    return _dedupe(issues)


def summary(issues: list[Issue]) -> dict[str, int]:
    out = {"total": len(issues), "spelling": 0, "grammar": 0}
    for issue in issues:
        out[issue.category] = out.get(issue.category, 0) + 1
    return out


if __name__ == "__main__":
    import sys

    sample = " ".join(sys.argv[1:]) or (
        "he go to the school evry day and dont never do he's homework . "
        "the studets was very happy ."
    )
    print(f"TEXT: {sample}\n")
    for it in analyze(sample):
        print(f"  [{it.category:8}] {it.rule_id:24} {it.text!r:20} -> "
              f"{it.suggestions}  ({it.message})")
