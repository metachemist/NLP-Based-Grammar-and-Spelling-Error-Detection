# NLP-Based Grammar and Spelling Error Detection

**University of Karachi — Department of Computer Science**

**Course:** Natural Language Processing

**Project Report**

**Submitted by:**

| Name | Seat No. |
|---|---|
| Shehryar Ahmed | Eb23210006123 |
| Hamza Owais | Eb23210006028 |
| Hafsa Shahid | Eb23210006026 |
| Syed Muhammad Jawad | Eb23210006135 |

---

## Abstract

This project is a program that reads English text and points out spelling and
grammar mistakes. It works in **four steps, stacked on top of each other**:

1. **Split** the text into sentences and words.
2. **Spelling check** — find words that are not real English words and suggest
   the closest correct word.
3. **Grammar rules** — a set of small, hand-written checks (for example: "a
   singular subject needs a singular verb").
4. **AI model (optional)** — a neural network that rewrites the sentence
   correctly; we compare its rewrite to the original to see what it changed.

The results are shown in a simple web page that highlights each mistake.

To measure how well it works, we built a test set of 36 short sentences: 30 with
exactly one planted mistake each, and 6 that are already correct. Using steps
1–3 only, the program finds **77%** of the planted mistakes. Adding the AI model
in step 4 raises this to **97%**. In both cases it raises **zero** false alarms
on the 6 correct sentences.

---

## 1. Introduction

### 1.1 Why this is useful

Tools like Grammarly, Microsoft Editor and LanguageTool are some of the most
popular uses of language technology. They have to handle two different kinds of
problem:

- **Spelling mistakes** — the word itself is wrong: *"recieve"*, *"teh"*,
  *"evry"*. These are easy to spot because the word does not exist.
- **Grammar mistakes** — every word is a real word, but they do not fit
  together: *"He go to school"*, *"a apple"*, *"I didn't knew"*. These are
  harder because you have to look at how the words relate to each other.

This project builds a small system that handles both.

### 1.2 What the system does

1. Find misspelled words and suggest corrections.
2. Find common grammar mistakes: subject–verb agreement, *a* vs *an*, repeated
   words, missing capital letters, spacing around punctuation, and wrong verb
   form after *do/does/did*.
3. Find harder, meaning-based mistakes (wrong preposition, wrong tense, *their*
   vs *there*) using an AI model.
4. Show everything in a usable interface, and measure the accuracy with numbers.

### 1.3 What is left out

- English only.
- It looks at one sentence at a time, not the whole document.
- It **detects** mistakes and gives one best suggestion; it does not silently
  rewrite your text.
- No personalisation, no style or tone advice.

---

## 2. Background: the ideas we use

This section explains the concepts behind each step, with an example for each.

### 2.1 Splitting text into sentences and words (tokenization)

Before anything else, raw text has to be broken into **sentences**, then into
**tokens** (words and punctuation marks). This is not just "split on spaces":

- *"don't"* becomes two tokens, *do* + *n't*.
- *"U.S.A."* is one token, not three sentences.
- The full stop in *"Dr. Khan"* does not end a sentence.

We use the **spaCy** library for this. Every token also remembers its position
(character number) in the original text, so we can highlight the exact spot
later.

### 2.2 Spelling correction by "closest real word" (the noisy-channel idea)

Think of a typo as a correct word that got **damaged** on its way to the page.
To fix *"evry"*, we ask: *which real word is this most likely to be a damaged
version of?*

Two things decide the answer:

- **How common the candidate word is.** *"every"* and *"very"* are both common;
  *"eery"* is rare.
- **How much damage it takes to get there.** We count single-character edits —
  insert a letter, delete a letter, swap a letter, or switch two neighbouring
  letters. This count is called the **edit distance**. *"evry" → "every"* is
  one insertion (distance 1).

We list every dictionary word within **edit distance 2** of the typo, then sort
those candidates by how common they are. This is Norvig's classic spelling
algorithm, provided by the `pyspellchecker` library.

Example: *"evry"* → suggestions *very, every, eery* (in that order, because
*very* is the most frequent word among the close matches).

### 2.3 Labelling each word's job (part-of-speech tagging)

A **part-of-speech (POS) tag** says what grammatical role a word plays: noun,
verb, adjective, and so on. spaCy uses fine-grained tags, for example:

| Tag | Meaning | Example |
|---|---|---|
| `VBZ` | verb, 3rd-person singular, present | *goes*, *is*, *has* |
| `VBP` | verb, present, not 3rd-person singular | *go*, *are*, *have* |
| `NN` | singular noun | *dog* |
| `NNS` | plural noun | *dogs* |

Knowing the tag lets a rule reason about **number** (singular/plural) and
**tense**.

### 2.4 Finding who does what (dependency parsing)

A **dependency parse** links each word to the word it depends on, with a label
for the relationship. The key label for us is `nsubj` — "nominal subject".

In *"The dog runs"*, the parse says *dog* is the `nsubj` of *runs*. That is
exactly the information a subject–verb agreement check needs: it can now compare
"is *dog* singular or plural?" with "is *runs* a singular or plural verb form?"

### 2.5 Grammar correction as translation (the AI model)

Modern AI grammar tools treat correction like **translation**: the input
language is "English with mistakes" and the output language is "correct
English". The model is a **Transformer** (the same family as translation and
chat models) called **T5-base**, already fine-tuned on thousands of
(wrong sentence, fixed sentence) pairs.

We give it a sentence, it gives back a corrected sentence, and we **compare the
two word by word**. Every word it changed, removed, or added is treated as a
detected mistake.

Strength: it catches mistakes nobody wrote a rule for. Weakness: it is slow, it
needs an 850 MB download, and it cannot explain *why* — the only "reason" is the
difference between the two sentences.

### 2.6 How we measure accuracy

- **Precision** = of all the things we flagged, how many were real mistakes.
- **Recall** = of all the real mistakes, how many we flagged.
- **F1** = a single score that balances the two.

Hand-written rules are usually high precision (few false alarms) but low recall
(they only know what we taught them). The AI model is the opposite.

---

## 3. System Design

### 3.1 The pipeline

```
raw text
   │
   ▼
[Layer 1]  spaCy: split into sentences + words, add POS tags and dependency parse
   │
   ├──────────────┬──────────────────────┐
   ▼              ▼                      ▼
[Layer 2]      [Layer 3]             [Layer 4]  (optional)
spelling       grammar rules         AI model (T5):
check          (8 rules over          rewrite sentence,
(edit          POS tags + parse)      diff against original
 distance)
   │              │                      │
   └──────────────┴──────────────────────┘
                  ▼
        merge + remove duplicates
        (keep the most specific finding for each spot)
                  ▼
        one sorted list of "Issue" records
                  ▼
        Streamlit web page: highlight each mistake
```

Layers 2 and 3 run offline in a few milliseconds. Layer 4 is behind a checkbox
because it is heavy.

### 3.2 One shared record type: `Issue`

Every layer, no matter how different, produces the **same** kind of record so
the rest of the program only has to understand one thing:

| Field | Meaning |
|---|---|
| `start`, `end` | character positions of the mistake in the original text |
| `text` | the exact wrong substring |
| `category` | `"spelling"` or `"grammar"` (used for colour) |
| `rule_id` | which check found it, e.g. `NON_WORD`, `SUBJECT_VERB_AGREEMENT` |
| `message` | plain-English explanation for the user |
| `suggestions` | best-first list of replacements (can be empty) |
| `source` | which layer: `spellchecker`, `rules`, or `neural` |

### 3.3 Layer 2 — spelling

For each word token, we **skip** it if it is:

- not purely alphabetic (numbers, punctuation, hyphenated words);
- a contraction fragment (*n't*, *'s*, *'re*, …);
- a single letter;
- a URL or email address;
- ALL CAPS (assumed to be an acronym like *NASA*, *HTTP*);
- tagged as a proper noun (names and places are not in a general dictionary).

Everything else is looked up in the dictionary. If it is missing, we build the
list of candidates within edit distance 2, sort them by frequency, keep the top
5, and re-apply the original capitalisation. The finding gets
`rule_id = NON_WORD`.

### 3.4 Layer 3 — the 8 grammar rules

Each rule is a small function that reads POS tags and the dependency parse.
Rules are deliberately **cautious**: they stay quiet unless they are fairly
sure, so they almost never raise a false alarm.

| `rule_id` | What it checks | Example it catches |
|---|---|---|
| `SENTENCE_CAPITALIZATION` | sentence starts with a lowercase letter | *"he went home."* |
| `LOWERCASE_I` | the word *i* used as the pronoun | *"then i left"* |
| `A_VS_AN` | *a* / *an* chosen by the next word's sound | *"a apple"*, *"an dog"* |
| `SUBJECT_VERB_AGREEMENT` | singular/plural subject vs. verb form | *"He go home"*, *"The children was"* |
| `DO_SUPPORT_BASE_FORM` | after *do/does/did*, the verb must be the plain form | *"I didn't knew"* → *know* |
| `REPEATED_WORD` | the same word twice in a row | *"the the plan"* |
| `SPACE_BEFORE_PUNCT` | a space before `, . ! ? ; :` | *"Hello ."* |
| `MISSING_SENTENCE_PUNCT` | a full sentence with no end mark | *"He went home"* (no full stop) |

### 3.5 Layer 4 — the AI model

Each sentence is sent to `vennify/t5-base-grammar-correction` (a T5-base model,
about 850 MB) with the prefix `"grammar: "`. The model returns a corrected
sentence. We line up the original and the correction **word by word** using
Python's `difflib`, and turn every change into an `Issue` with
`rule_id = NEURAL_GEC`:

- word replaced or deleted → a normal highlight over those words;
- word inserted → a zero-width marker between two words (shown as `⋯`).

Two practical notes:

- We call the model directly (`AutoModelForSeq2SeqLM.generate`, 4-beam search)
  instead of the `transformers` "pipeline" helper, because the helper's
  text-to-text task was removed in version 5 of the library.
- The environment variable `NEURAL_GEC_MODEL` can point at a local folder, so
  the model can run with no internet.

### 3.6 Merging the three layers

All findings are collected, then sorted so that the **more specific** layer wins
when two findings cover the same spot. The priority is:

> spelling (most specific) > rules > AI model (most generic)

If a finding overlaps one that is already kept from a higher-priority layer, it
is dropped. So a precise *"'evry' is misspelled"* beats a vague *"the model
changed this word"*. Zero-width AI insertions are always kept, because they
never really overlap anything.

### 3.7 The interface

A **Streamlit** web page:

- **Left sidebar:** a short explanation of the four layers and a checkbox to
  turn Layer 4 on (greyed out if the AI libraries are not installed).
- **Main area:** a text box, three counters (total / spelling / grammar), the
  text itself with coloured highlights (red = spelling, blue = grammar) and a
  tooltip on each, and a list of every issue with its suggestions.

---

## 4. Implementation

- **Language and libraries:** Python 3.12, spaCy 3.8 (`en_core_web_sm` model),
  `pyspellchecker`, Streamlit. Layer 4 additionally needs `transformers` 5 and
  `torch` 2 (the CPU build is enough).
- **Size:** about 600 lines of code across `src/` (around 780 counting blank
  lines and comments).
- **Design decisions worth noting:**
  - one shared `Issue` type, so merging and scoring are simple;
  - Layer 4 is written so that the program still imports and runs on a machine
    with no AI libraries installed;
  - spaCy is loaded once and reused (cached), because loading it is the slow
    part;
  - the scoring code is completely separate from the detection code.

---

## 5. Evaluation

### 5.1 The test set

`data/eval_pairs.jsonl` contains **36 short sentences**:

- **30 "error" sentences**, each with exactly one planted mistake, grouped into
  **13 categories** (misspelling, *a*/*an*, subject–verb, wrong tense, wrong
  preposition, *their/there*, and so on);
- **6 "clean" sentences** that are already correct, used to check for false
  alarms.

Each error sentence is stored together with its corrected version, so the
program knows exactly where the mistake is.

### 5.2 What we measure

- **Detection recall** — the error sentence produced at least one flag.
- **Localization rate** — at least one flag actually lands on the words that
  differ between the wrong and correct versions (not just somewhere in the
  sentence).
- **False-positive rate** — a clean sentence produced any flag at all (we want
  this to be 0).

### 5.3 Results

| Configuration | Detection recall | Localization | False alarms (clean) |
|---|---|---|---|
| Layers 2–3 (offline rules only) | **77%** (23/30) | 77% (23/30) | **0%** (0/6) |
| Layers 2–4 (with the AI model) | **97%** (29/30) | 97% (29/30) | **0%** (0/6) |

Broken down by mistake type (detection / localization):

| Mistake type | Count | Rules only | With AI model |
|---|--:|--:|--:|
| a vs an | 3 | 100% / 100% | 100% / 100% |
| do/did + wrong verb form | 3 | 100% / 100% | 100% / 100% |
| lowercase "i" | 2 | 100% / 100% | 100% / 100% |
| missing end punctuation | 1 | 100% / 100% | 100% / 100% |
| misspelled word | 4 | 100% / 100% | 100% / 100% |
| repeated word | 2 | 100% / 100% | 100% / 100% |
| missing capital letter | 2 | 100% / 100% | 100% / 100% |
| space before punctuation | 2 | 100% / 100% | 100% / 100% |
| subject–verb agreement | 5 | 80% / 80% | 100% / 100% |
| extra word | 1 | 0% / 0% | 100% / 100% |
| wrong preposition | 2 | 0% / 0% | 50% / 50% |
| wrong word (their/there) | 1 | 0% / 0% | 100% / 100% |
| wrong verb tense | 2 | 0% / 0% | 100% / 100% |

**Reading the rules-only column.** The rules score 100% on everything they were
built for. The single subject–verb miss is *"The dog chase the ball…"*: spaCy
mislabels *chase* as a noun, so the agreement rule never runs. The four
categories at 0% — extra words, prepositions, *their/there*, and tense — are
exactly the meaning-based mistakes that rules cannot see.

**What the AI model adds.** Overall detection jumps from 77% to 97%. It fixes
every rule blind spot except one of the two preposition cases (*"interested on
learning"* → *in* is caught; *"good in mathematics"* → *at* is missed), and it
also handles the *"dog chase"* sentence that broke the parser. Importantly, it
raised **no** false alarms on the 6 clean sentences — it does not "correct"
text that is already fine.

**The cost.** The AI run took about **1 minute 52 seconds** for 30 sentences on
a normal CPU (roughly 3.7 seconds per sentence). The rule layers finish all 30
in well under a second.

### 5.4 Discussion

**What the rules do well.** Misspellings, *a/an*, lowercase *i*, repeated words,
capitalisation, punctuation spacing, *do*-support, and subject–verb agreement
when the subject is a clear pronoun or noun — all 100% here. These are **local**
patterns: one POS tag plus one link in the parse is enough to decide.

**What needs the AI model.** Prepositions, tense across a clause, *their/there*,
and missing/extra words. The rules get 0% on all four; the model recovers three
of them fully and one preposition case out of two. These mistakes depend on
**meaning**, which a tag pattern cannot capture.

**About the "0% false alarms".** This is reassuring but it is only 6 easy
sentences. Real text would eventually produce false alarms — likely from unusual
proper nouns, commands read as sentences missing a subject, or informal writing
that is not actually wrong.

**Rules vs. AI, in one line.** Rules are instant and every flag comes with a
named reason; the AI model roughly quadruples what we catch but is slow, large,
and cannot explain itself.

---

## 6. Limitations and future work

**Current limitations**

- Without the AI model, wrong-word mistakes like *form* vs *from* are invisible.
- The spell checker skips hyphenated and compound words.
- The rules are specific to English and to formal writing.
- The AI model sometimes rephrases a sentence instead of making the smallest
  fix, which makes the highlighted region bigger than it should be.

**Possible extensions**

- A confidence score on each flag.
- A second AI layer (a masked language model such as BERT) aimed at wrong-word
  mistakes.
- Training on a properly labelled dataset (for example W&I+LOCNESS) and scoring
  with the standard **ERRANT** tool instead of our own metric.
- Running the AI model on many sentences at once for speed.

---

## 7. Conclusion

The system stacks four layers: a "closest real word" spell checker, a set of
cautious grammar rules, and an optional AI grammar model, all feeding one shared
list of findings shown in a web page. Splitting the work this way makes the
trade-off obvious — rules give precision, the AI model gives coverage — and lets
the program still run usefully on a machine that cannot load the AI model.

---

## References

1. P. Norvig, *How to Write a Spelling Corrector*, 2007.
   https://norvig.com/spell-correct.html
2. D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed.
   (draft) — chapters on spelling correction and POS tagging.
3. spaCy documentation — https://spacy.io
4. C. Bryant et al., *The BEA-2019 Shared Task on Grammatical Error Correction*.
5. C. Napoles et al., *JFLEG: A Fluency Corpus and Benchmark for GEC*, 2017.
6. C. Raffel et al., *Exploring the Limits of Transfer Learning with a Unified
   Text-to-Text Transformer* (T5), 2020.
7. `pyspellchecker` — https://github.com/barrust/pyspellchecker

---

## Appendix A: Installation and usage

### A.1 Setup

```bash
cd "NLP-Based Grammar and Spelling Error Detection"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Optional: enable Layer 4 (the AI model). CPU-only torch keeps it small.
pip install "transformers>=4.40" sentencepiece
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

To run Layer 4 offline, download the model once and point `NEURAL_GEC_MODEL` at
the folder:

```bash
hf download vennify/t5-base-grammar-correction --local-dir models/t5-base-grammar-correction
export NEURAL_GEC_MODEL=models/t5-base-grammar-correction
```

### A.2 Running

```bash
# Web app
streamlit run app.py

# One sentence on the command line
python -m src.pipeline "he go to school evry day"

# Accuracy on the test set
python -m src.evaluate            # Layers 2–3
python -m src.evaluate --neural   # add Layer 4
```

---

## Appendix B: Repository structure

The code is a small package with **one module per layer**, matching Section 3.

```
src/
  issue.py           the Issue record — shared output of every layer
  nlp_core.py        loads spaCy once (splitter + POS + parser)
  spell_checker.py   Layer 2 — spelling
  grammar_rules.py   Layer 3 — the 8 grammar rules (one function each)
  pipeline.py        runs the layers and merges the findings
  neural_gec.py      Layer 4 — the AI model (safe to import without it installed)
  evaluate.py        Layer 5 — scoring on the test set
data/
  eval_pairs.jsonl   the 36 test sentences (wrong, correct, mistake type)
app.py               the Streamlit web page
report/report.md     this report
```
