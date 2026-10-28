"""Streamlit tab for Module 1."""

from __future__ import annotations

import json

import streamlit as st

from lawlens.summariser import ROLES, ROLE_TITLES, load_document, summarise, verify_citations
from lawlens.summariser.ingest import load_text


@st.cache_resource(show_spinner="Loading sentence-embedding model (first run downloads ~90 MB)...")
def _embedder():
    from lawlens.summariser.embedder import Embedder
    return Embedder()


@st.cache_resource(show_spinner="Loading DistilBART (first run downloads ~1.2 GB)...")
def _compressor():
    from lawlens.summariser.abstractive import AbstractiveCompressor
    return AbstractiveCompressor()


def _esc(s: str) -> str:
    return s.replace("$", r"\$")


def _highlight(para_text: str, sentences: list[str]) -> str:
    out = _esc(para_text)
    for s in sentences:
        s = s[:-1].rstrip() if s.endswith("…") else s
        if s and _esc(s) in out:
            out = out.replace(_esc(s), f"**{_esc(s)}**", 1)
    return out


def render():
    st.subheader("Judgment summariser")
    st.caption(
        "Upload an Indian court judgment (text-based PDF or .txt) or paste its text. "
        "Every point is tied to a paragraph so you can check it against the source."
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        up = st.file_uploader("Judgment (PDF or TXT)", type=["pdf", "txt"], key="m1_upload")
        pasted = st.text_area("...or paste text", height=120, key="m1_paste")
    with c2:
        max_words = st.slider("Summary length (words)", 200, 1000, 450, 50, key="m1_len")
        use_emb = st.checkbox("Experimental: sentence-embedding model (no measured gain, slower)",
                              value=False, key="m1_emb")
        use_abs = st.checkbox("Abstractive rewrite (slower, may paraphrase)", value=False, key="m1_abs")

    if st.button("Summarise", type="primary", key="m1_go"):
        try:
            if up is not None:
                paras = load_document(up.name, up.getvalue())
            elif pasted.strip():
                paras = load_text(pasted)
            else:
                st.warning("Upload a file or paste some text first.")
                return
            with st.spinner("Summarising..."):
                summary = summarise(
                    paras,
                    _embedder() if use_emb else None,
                    max_words=max_words,
                    compressor=_compressor() if use_abs else None,
                )
            st.session_state["m1"] = (paras, summary)
        except Exception as e:  # show a readable message rather than a stack trace
            st.error(str(e))
            return

    if "m1" not in st.session_state:
        return
    paras, summary = st.session_state["m1"]
    by_idx = {p.idx: p for p in paras}
    check = verify_citations(summary, paras)

    m1, m2, m3 = st.columns(3)
    m1.metric("Paragraphs in document", summary.n_paragraphs)
    m2.metric("Summary words", summary.n_words())
    m3.metric("Quotes verified in source", f"{check['verified']}/{check['checked']}" if check["checked"] else "n/a (paraphrased)")

    for role in ROLES:
        st.markdown(f"### {ROLE_TITLES[role]}")
        pts = summary.by_role()[role]
        if not pts:
            st.info("Not clearly identified in this document.")
        for i, p in enumerate(pts):
            tag = "" if p.verbatim else " *(paraphrased)*"
            tag += "" if p.cue_hit else " ⚠️ *low confidence: no legal cue phrase, chosen by position*"
            st.markdown(f"- {_esc(p.text)} **[¶{p.label}]**{tag}")
            with st.expander(f"Source paragraph ¶{p.label}"):
                st.markdown(_highlight(by_idx[p.para_idx].text, p.sentences if p.verbatim else []))

    st.divider()
    d1, d2 = st.columns(2)
    d1.download_button("Download as Markdown", summary.to_markdown(), "summary.md", key="m1_dl_md")
    d2.download_button("Download as JSON", json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
                       "summary.json", key="m1_dl_json")
    st.caption(
        "Decision-support prototype, not legal advice. Roles are found with zero-shot heuristics and "
        "can be wrong: always read the cited paragraph."
    )
