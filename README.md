# NLP-Based Grammar and Spelling Error Detection

A layered NLP pipeline that flags **spelling** and **grammar** mistakes in
English text and highlights them in a web app.

| Layer | Technique | NLP concept | Speed |
|------:|-----------|-------------|-------|
| 1 | Tokenization + sentence segmentation (spaCy) | text preprocessing | instant |
| 2 | Dictionary + edit distance (`pyspellchecker`) | noisy-channel model, Levenshtein distance | instant |
| 3 | ~7 hand-written rules over POS tags + dependency parse | morphology, POS tagging, syntactic parsing | instant |
| 4 | T5 seq2seq grammar correction, diffed against the input *(optional)* | Transformers, grammatical error correction (GEC) | slow, ~850 MB |

Layers 2–3 run offline in milliseconds. Layer 4 is behind a checkbox and the
app works without it.

## Setup

```bash
cd "NLP-Based Grammar and Spelling Error Detection"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Optional: enable Layer 4 (the Transformer). CPU-only torch keeps the
# download small; drop the --index-url for the default (CUDA) build.
pip install "transformers>=4.40" sentencepiece
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The T5 model (`vennify/t5-base-grammar-correction`, ~850 MB) downloads from the
Hugging Face Hub on first use. To run Layer 4 offline, pre-download it and point
`NEURAL_GEC_MODEL` at a local directory:

```bash
hf download vennify/t5-base-grammar-correction --local-dir models/t5-base-grammar-correction
export NEURAL_GEC_MODEL=models/t5-base-grammar-correction   # HF id or local path
```

## Run

```bash
# Web app
streamlit run app.py

# Command-line, single sentence
python -m src.pipeline "he go to school evry day"

# Evaluation on the minimal-pair test set
python -m src.evaluate            # Layers 2-3
python -m src.evaluate --neural   # + Layer 4
```

## Project layout

```
src/
  issue.py         Issue dataclass -- the common output of every layer
  nlp_core.py      loads spaCy once (tokenizer + POS + parser)
  spell_checker.py Layer 2
  grammar_rules.py Layer 3  (one function per rule)
  neural_gec.py    Layer 4  (import-safe without transformers)
  pipeline.py      runs the layers, deduplicates overlapping findings
  evaluate.py      Layer 5  (metrics)
data/
  eval_pairs.jsonl  36 (bad, good, phenomenon) rows: 30 errors + 6 clean
app.py             Streamlit UI
report/report.md   written report
```

## How the pieces map to NLP theory

- **Tokenization** — turning a character string into linguistic units. Non-trivial
  because of punctuation, contractions ("don't" → "do" + "n't"), and abbreviations.
- **Edit distance / noisy-channel model** — a misspelling is a "corruption" of an
  intended word through a noisy channel; the best correction maximises
  `P(word) · P(typo | word)`, approximated by "closest real word, weighted by
  frequency".
- **POS tagging** — assigning `VBZ` / `NNS` / … to each token with a statistical
  sequence model; lets rules reason about number and tense.
- **Dependency parsing** — a tree linking each word to its syntactic head; lets a
  rule find *which* noun is the subject of *which* verb.
- **Grammatical Error Correction (GEC)** — framed as machine translation from
  "bad English" to "good English"; a Transformer trained on millions of such
  pairs generalises far past hand-written rules, at the cost of interpretability.
- **Evaluation** — precision / recall / F1; the recurring trade-off is that rules
  are precise but narrow while the neural model is broad but noisier.
