"""The single data structure every detection layer produces.

Keeping one shared type means the pipeline can merge spelling errors,
rule-based grammar errors, and neural grammar errors into one list and
the UI only has to know how to render *this*.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Issue:
    # Character offsets into the ORIGINAL text, so the UI can highlight it.
    start: int
    end: int
    # The exact substring that is (probably) wrong.
    text: str
    # "spelling" or "grammar" -- used for colour-coding in the UI.
    category: str
    # A stable machine ID for the specific check, e.g. "NON_WORD",
    # "SUBJECT_VERB_AGREEMENT", "A_VS_AN", "NEURAL_GEC". Handy for evaluation
    # breakdowns ("how well do we do on subject-verb agreement?").
    rule_id: str
    # Human-readable explanation shown to the user.
    message: str
    # Ordered best-first list of suggested replacements (may be empty).
    suggestions: list[str] = field(default_factory=list)
    # Which layer found it: "spellchecker" | "rules" | "neural".
    source: str = ""

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    def overlaps(self, other: "Issue") -> bool:
        return self.start < other.end and other.start < self.end
