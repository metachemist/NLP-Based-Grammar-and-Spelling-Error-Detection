"""Streamlit web app: paste text, see spelling & grammar errors highlighted.

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import html as _html

import streamlit as st

from src import neural_gec
from src.pipeline import analyze, summary

_SAMPLE = (
    "he go to the school evry day and dont never do he's homework. "
    "the studetns was very happy about a apple they recieved yesterday . "
    "i think this is definitly a interesting a interesting result"
)

_COLOURS = {"spelling": "#ffd9d9", "grammar": "#d6e4ff"}


def render_html(text: str, issues) -> str:
    issues = sorted(issues, key=lambda i: (i.start, i.end))
    parts, cursor = [], 0
    for it in issues:
        if it.start < cursor:
            continue  # safety net for any residual overlap
        parts.append(_html.escape(text[cursor:it.start]))
        tip = _html.escape(it.message)
        if it.start == it.end:  # a neural "insert here" marker
            parts.append(f'<span title="{tip}" '
                         f'style="color:#c026d3;font-weight:700">&#8943;</span>')
        else:
            seg = _html.escape(text[it.start:it.end])
            colour = _COLOURS.get(it.category, "#eee")
            parts.append(f'<mark title="{tip}" style="background:{colour};'
                         f'padding:0 2px;border-radius:3px">{seg}</mark>')
        cursor = max(cursor, it.end)
    parts.append(_html.escape(text[cursor:]))
    return "".join(parts).replace("\n", "<br>")


st.set_page_config(page_title="Grammar & Spelling Error Detection", layout="wide")
st.title("NLP-Based Grammar & Spelling Error Detection")
st.caption("A layered pipeline: dictionary + edit distance, POS/dependency rules, "
           "and an optional Transformer grammar model.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "**Layer 1 – Tokenization** (spaCy): split into sentences & tokens.\n\n"
        "**Layer 2 – Spelling**: word not in dictionary → nearest word by "
        "edit distance, ranked by frequency (noisy-channel model).\n\n"
        "**Layer 3 – Grammar rules**: POS tags + dependency parse feed ~7 "
        "explicit rules (subject–verb agreement, a/an, repeated words, …).\n\n"
        "**Layer 4 – Neural GEC** *(optional)*: a T5 model rewrites the "
        "sentence; the diff against your text is the set of errors."
    )
    st.divider()
    neural_ok = neural_gec.is_available()
    use_neural = st.checkbox(
        "Enable Layer 4 (Transformer)", value=False, disabled=not neural_ok,
        help="Downloads ~850 MB on first run.",
    )
    if not neural_ok:
        st.caption("Install `transformers` + `torch` to enable Layer 4.")

text = st.text_area("Text to check", value=_SAMPLE, height=160)

if st.button("Analyze", type="primary"):
    with st.spinner("Analyzing…"):
        issues = analyze(text, use_neural=use_neural)
    stats = summary(issues)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total issues", stats["total"])
    c2.metric("Spelling", stats["spelling"])
    c3.metric("Grammar", stats["grammar"])

    st.subheader("Annotated text")
    st.markdown(
        f'<div style="line-height:2;font-size:1.05rem;border:1px solid #ddd;'
        f'border-radius:8px;padding:14px">{render_html(text, issues)}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Issues")
    if not issues:
        st.success("No issues found.")
    for it in issues:
        sug = ", ".join(f"`{s}`" for s in it.suggestions) or "—"
        st.markdown(
            f"- **{it.category}** · `{it.rule_id}` · _{it.source}_  \n"
            f"  {it.message}  \n"
            f"  Suggestion(s): {sug}"
        )
