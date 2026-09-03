# NLP-Based Grammar and Spelling Error Detection

**Author:** _<your name>_
**Course / Assignment:** _<course code>_
**Date:** _<date>_

---

## Abstract

_(Write last, ~150 words.)_ This project builds a system that automatically
detects spelling and grammatical errors in English text. It uses a **layered
pipeline**: (1) tokenization, (2) a noisy-channel spelling corrector based on
edit distance, (3) a set of rules operating on part-of-speech tags and a
dependency parse, and (4) an optional Transformer-based grammatical error
correction (GEC) model. The layers are combined and deduplicated, and results
are shown in an interactive web application. On a 32-item minimal-pair test set
the system detects _XX%_ of introduced errors with a _YY%_ false-positive rate
on clean sentences.

## 1. Introduction

### 1.1 Motivation
Writing assistants (Grammarly, Microsoft Editor, LanguageTool) are among the
most widely used NLP applications. They must solve two related but distinct
problems: catching **misspellings** (a token that is not a valid word) and
catching **grammatical errors** (all tokens valid, but the combination is
wrong).

### 1.2 Objectives
1. Detect non-word spelling errors and propose corrections.
2. Detect common grammatical errors (subject–verb agreement, article choice,
   repeated words, capitalization, punctuation spacing).
3. Detect context-sensitive errors that rules miss, using a neural model.
4. Present results through a usable interface and evaluate them quantitatively.

### 1.3 Scope and limitations
English only; sentence-level analysis; detection and single-best suggestion
(not full auto-rewriting); no user personalization or style checking.

## 2. Background / Related Concepts

### 2.1 Tokenization and sentence segmentation
Splitting raw text into sentences and then tokens (words, punctuation,
sub-word clitics such as *n't*). All later stages operate on these units and on
their character offsets in the original string.

### 2.2 The noisy-channel model for spelling correction
A misspelling `x` is modelled as an intended word `w` passed through a noisy
channel. The best correction is
`ŵ = argmax_w P(w) · P(x | w)`,
where `P(w)` is a word-frequency prior and `P(x | w)` decreases with the number
of single-character edits (insertion, deletion, substitution, transposition —
the **Damerau–Levenshtein distance**) between `x` and `w`. In practice we
enumerate all dictionary words within edit distance ≤ 2 and rank them by
frequency (Norvig's algorithm, as implemented by `pyspellchecker`).

### 2.3 Part-of-speech tagging
A statistical sequence model assigns each token a grammatical category. We use
spaCy's fine-grained Penn-Treebank tags, e.g. `VBZ` (3rd-person-singular present
verb), `VBP` (other present verb), `NN` / `NNS` (singular / plural noun). These
let a rule reason about **number** and **tense**.

### 2.4 Dependency parsing
A dependency parse is a tree in which every word points to its syntactic
**head** with a labelled relation (`nsubj`, `dobj`, …). It answers "which noun
is the subject of which verb?", which is exactly what a subject–verb agreement
check needs.

### 2.5 Grammatical Error Correction (GEC) as sequence-to-sequence
Modern GEC treats correction as translation from *ungrammatical* to
*grammatical* English. A Transformer encoder–decoder (here **T5-base**,
fine-tuned on the JFLEG / C4-200M-style corpora) is trained on millions of
(source, target) pairs. It generalises to error types never explicitly
programmed, at the cost of interpretability and speed.

### 2.6 Evaluation metrics
- **Precision** = correct flags / all flags.
- **Recall** = correct flags / all true errors.
- **F1** = harmonic mean.
Rule-based components favour precision; neural components favour recall.

## 3. System Design

### 3.1 Architecture

```
raw text
   │
spaCy: tokenize + sentence-split + POS + dependency parse      (Layer 1)
   │
   ├── spell_checker.check_spelling   dictionary + edit distance   (Layer 2)
   ├── grammar_rules.check_grammar    7 rules over tags + parse     (Layer 3)
   └── neural_gec.neural_issues       T5 rewrite → word-level diff  (Layer 4, optional)
   │
pipeline._dedupe   drop overlapping findings, keep the most specific
   │
sorted list of Issue(start, end, category, rule_id, message, suggestions, source)
   │
Streamlit UI: inline highlights + issue list
```

### 3.2 The `Issue` data structure
Every layer emits the same record type (character span, category, stable
`rule_id`, human message, ranked suggestions, source layer). This is what makes
merging and evaluation uniform.

### 3.3 Layer 2 — spelling
For each alphabetic token that is not a proper noun, an acronym, a URL/email, or
a known dictionary word: enumerate candidates within edit distance 2, rank by
corpus frequency, keep the top 5, preserve original casing.

### 3.4 Layer 3 — grammar rules
| `rule_id` | What it checks | Signals used |
|---|---|---|
| `SENTENCE_CAPITALIZATION` | first token of a sentence is lower-case | `doc.sents` |
| `LOWERCASE_I` | standalone pronoun "i" | token text |
| `A_VS_AN` | "a"/"an" vs. following word's initial sound | next token + exception lists |
| `SUBJECT_VERB_AGREEMENT` | singular/plural subject vs. `VBZ`/`VBP` verb | `nsubj` dependency, `tag_` |
| `REPEATED_WORD` | same word twice in a row | adjacent tokens |
| `SPACE_BEFORE_PUNCT` | whitespace before `, . ! ? ; :` | regex on raw text |
| `MISSING_SENTENCE_PUNCT` | clause with a finite verb, no terminal punctuation | `doc.sents`, `tag_` |

### 3.5 Layer 4 — neural GEC
Each sentence is prefixed with `"grammar: "` and passed to
`vennify/t5-base-grammar-correction`. The output is aligned to the input with a
word-level `difflib` diff; every `replace` / `delete` / `insert` opcode becomes
an `Issue` with `rule_id = NEURAL_GEC`, mapped back to character offsets.

### 3.6 Merging
Findings are sorted by (source priority, position). A finding is dropped if it
overlaps an already-kept finding from a higher-priority source
(spelling > rules > neural), so a specific "misspelled word" beats a generic
"the model changed this".

### 3.7 Interface
Streamlit. Left sidebar explains the four layers and toggles Layer 4. Main pane:
a text box, per-category counts, the text with `<mark>` highlights (red =
spelling, blue = grammar) and hover tooltips, and a list of issues with
suggestions.

## 4. Implementation

- **Language / libraries:** Python 3.12, spaCy 3.x (`en_core_web_sm`),
  `pyspellchecker`, Streamlit; optionally `transformers` + `torch`.
- **Lines of code:** ~_XXX_ across `src/`.
- **Design choices worth noting:**
  - one shared `Issue` type;
  - Layer 4 import-safe so the project runs on machines without PyTorch;
  - spaCy loaded once via `lru_cache`;
  - evaluation kept separate from the pipeline.

## 5. Evaluation

### 5.1 Test set
`data/eval_pairs.jsonl` — 32 minimal pairs: 27 `(bad, good)` sentences across 9
error phenomena, plus 5 clean sentences to measure false positives.

### 5.2 Metrics
- **Detection recall** — a `bad` sentence produced ≥ 1 issue.
- **Localization rate** — an issue overlapped the span that differs from `good`.
- **False-positive rate** — a clean sentence produced any issue.

### 5.3 Results

| Configuration | Detection recall | Localization | FP rate (clean) |
|---|---|---|---|
| Layers 2–3 (offline) | **74%** (20/27) | 74% | **0%** (0/5) |
| Layers 2–4 (+ T5) | _run `python -m src.evaluate --neural` and paste_ | _XX%_ | _XX%_ |

Per-phenomenon (Layers 2–3):

| phenomenon | n | detect | localize |
|---|--:|--:|--:|
| a_vs_an | 3 | 100% | 100% |
| lowercase_i | 2 | 100% | 100% |
| missing_sentence_punct | 1 | 100% | 100% |
| non_word_spelling | 4 | 100% | 100% |
| repeated_word | 2 | 100% | 100% |
| sentence_capitalization | 2 | 100% | 100% |
| space_before_punct | 2 | 100% | 100% |
| subject_verb | 5 | 80% | 80% |
| extra_word | 1 | 0% | 0% |
| preposition | 2 | 0% | 0% |
| real_word_confusion | 1 | 0% | 0% |
| verb_tense | 2 | 0% | 0% |

The rule layer handles every phenomenon it was designed for at 100%, plus 4/5
subject–verb cases (the miss is a spaCy parser error that tags the main verb
"chase" as a noun). The four phenomena at 0% — extra/missing words, preposition
choice, real-word confusion, verb tense — are precisely the context-sensitive
errors that rules cannot see and Layer 4 is meant to recover.

### 5.4 Discussion
- Which phenomena do the rules catch reliably? (spelling, a/an, repeated words,
  capitalization, subject–verb agreement with clear subjects.)
- Which need the neural layer? (preposition choice, verb tense across a clause,
  real-word confusions like *their/there*, missing/extra words.)
- Where do false positives come from? (proper nouns not in the dictionary,
  imperatives flagged as missing subjects, informal but valid style.)

## 6. Limitations and Future Work

- No real-word spelling errors without Layer 4 (*form* vs *from*).
- Hyphenated and compound tokens are skipped by the spell checker.
- Rules are English- and register-specific.
- The T5 model sometimes paraphrases rather than minimally correcting, inflating
  the diff.
- **Future:** confidence scores per issue; a masked-LM (BERT) layer for
  real-word errors; fine-tuning GECToR for token-level edits; a labelled
  span-level dataset (e.g. W&I+LOCNESS) with the standard **ERRANT** scorer;
  batching Layer 4 for speed.

## 7. Conclusion

A four-layer pipeline combining a noisy-channel spell checker, syntactic rules,
and an optional Transformer GEC model detects a broad range of English writing
errors. The layered design makes the trade-off between precision (rules) and
recall (neural) explicit and lets the system degrade gracefully when heavy
dependencies are unavailable.

## References

1. P. Norvig, "How to Write a Spelling Corrector", 2007.
   https://norvig.com/spell-correct.html
2. D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed. draft
   — ch. on spelling correction (noisy channel) and ch. on POS tagging.
3. spaCy documentation — https://spacy.io
4. Bryant et al., "The BEA-2019 Shared Task on Grammatical Error Correction".
5. Napoles et al., "JFLEG: A Fluency Corpus and Benchmark for GEC", 2017.
6. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified
   Text-to-Text Transformer" (T5), 2020.
7. `pyspellchecker` — https://github.com/barrust/pyspellchecker
