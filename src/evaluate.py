"""Layer 5 -- evaluation.

We use a minimal-pair test set (`data/eval_pairs.jsonl`): each item has a
`bad` sentence, its `good` correction, and a `phenomenon` label.

Metrics
-------
On the *bad* sentences (grouped by phenomenon):
  detection recall  = fraction where the pipeline raised >=1 issue
  localization rate = fraction where an issue actually overlaps the span
                      that differs between `bad` and `good`
On the *clean* sentences:
  false-positive rate = fraction where the pipeline raised any issue
                        (should be 0)

Run:
    python -m src.evaluate            # Layers 2-3 only (fast, offline)
    python -m src.evaluate --neural   # also Layer 4 (slow, needs transformers)
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import pathlib

from .pipeline import analyze

_DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "eval_pairs.jsonl"


def _changed_spans(bad: str, good: str) -> list[tuple[int, int]]:
    sm = difflib.SequenceMatcher(a=bad, b=good, autojunk=False)
    spans = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        spans.append((max(0, i1 - 1), i2 + 1))  # pad by 1 char for insertions
    return spans


def _localized(issues, spans) -> bool:
    return any(
        i.start <= e and s <= i.end
        for i in issues
        for (s, e) in spans
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural", action="store_true", help="include Layer 4 (T5)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in _DATA.read_text().splitlines() if line.strip()]

    by_phenom = collections.defaultdict(lambda: {"n": 0, "detected": 0, "localized": 0})
    clean_total = clean_fp = 0

    for row in rows:
        bad, good, phenom = row["bad"], row["good"], row["phenomenon"]

        if phenom == "clean":
            issues = analyze(good, use_neural=args.neural)
            clean_total += 1
            clean_fp += 1 if issues else 0
            continue

        issues = analyze(bad, use_neural=args.neural)
        bucket = by_phenom[phenom]
        bucket["n"] += 1
        if issues:
            bucket["detected"] += 1
        if _localized(issues, _changed_spans(bad, good)):
            bucket["localized"] += 1

    print(f"\nEvaluation ({'Layers 2-4' if args.neural else 'Layers 2-3'})")
    print("-" * 62)
    print(f"{'phenomenon':<24}{'n':>4}{'detect':>10}{'localize':>12}")
    print("-" * 62)
    tot_n = tot_d = tot_l = 0
    for phenom in sorted(by_phenom):
        b = by_phenom[phenom]
        tot_n += b["n"]; tot_d += b["detected"]; tot_l += b["localized"]
        print(f"{phenom:<24}{b['n']:>4}{b['detected'] / b['n']:>10.0%}"
              f"{b['localized'] / b['n']:>12.0%}")
    print("-" * 62)
    print(f"{'OVERALL':<24}{tot_n:>4}{tot_d / tot_n:>10.0%}{tot_l / tot_n:>12.0%}")
    print(f"\nFalse-positive rate on {clean_total} clean sentences: "
          f"{clean_fp / clean_total:.0%} ({clean_fp}/{clean_total})\n")


if __name__ == "__main__":
    main()
