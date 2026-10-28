"""Offline tests for Module 1: no model downloads. A hashing 'embedder' stands in for MiniLM."""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lawlens.summariser import load_document, summarise, verify_citations, ROLES
from lawlens.summariser.textutils import clean_text, split_sentences, chunk_text, split_paragraphs

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "sample_judgment.txt"


class HashEmbedder:
    """Deterministic bag-of-words hashing vectors, L2-normalised. Test double only."""

    def encode(self, texts, batch_size=64):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, hash(w) % 256] += 1.0
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


def _paras():
    return load_document("sample_judgment.txt", SAMPLE.read_bytes())


def test_numbered_paragraphs_keep_their_own_labels():
    paras = _paras()
    labels = [p.label for p in paras]
    assert labels[0] == "Header"
    assert labels[1:] == [str(i) for i in range(1, 13)]


def test_every_quoted_sentence_is_verbatim_in_cited_paragraph():
    paras = _paras()
    for emb in (None, HashEmbedder()):
        s = summarise(paras, emb, max_words=300)
        v = verify_citations(s, paras)
        assert v["checked"] > 0
        assert v["verified"] == v["checked"], v["failures"]


def test_decision_comes_from_the_final_paragraphs():
    paras = _paras()
    s = summarise(paras, None, max_words=300)
    dec = s.by_role()["decision"]
    assert dec and dec[0].label == "12"
    assert "appeal is allowed" in dec[0].text.lower()


def test_issues_and_arguments_found():
    s = summarise(_paras(), None, max_words=400)
    by = s.by_role()
    assert any(p.label == "4" for p in by["issues"])
    assert {"5", "6"} & {p.label for p in by["arguments"]}


def test_paragraph_is_cited_by_at_most_one_role():
    s = summarise(_paras(), HashEmbedder(), max_words=500)
    labels = [p.label for p in s.points]
    assert len(labels) == len(set(labels))


def test_word_budget_roughly_respected():
    s = summarise(_paras(), None, max_words=150)
    assert s.n_words() <= 150 * 1.6


def test_cue_heavy_weights_leave_absent_roles_empty():
    text = "\n".join(f"{i}. The weather was pleasant and the garden was green on that day." for i in range(1, 8))
    s = summarise(split_paragraphs(clean_text(text)), None, max_words=200, weights=(0.69, 0.0, 0.31))
    assert not s.by_role()["decision"]
    assert not s.by_role()["arguments"]


def test_frozen_weights_flag_position_only_points_as_low_confidence():
    """The validated weights are position-heavy, so non-legal text can still yield picks.
    They must be flagged rather than presented as confident."""
    text = "\n".join(f"{i}. The weather was pleasant and the garden was green on that day." for i in range(1, 8))
    s = summarise(split_paragraphs(clean_text(text)), None, max_words=200)
    assert all(not p.cue_hit for p in s.points)
    judgment = summarise(_paras(), None, max_words=300)
    dec = judgment.by_role()["decision"]
    assert dec and dec[0].cue_hit


def test_clean_text_drops_page_furniture_and_dehyphenates():
    raw = "The respon-\ndent filed.\nPage 3 of 20\n17\nIndian Kanoon - http://indiankanoon.org/doc/1/\nNext."
    out = clean_text(raw)
    assert "respondent" in out and "Page 3" not in out and "Kanoon" not in out
    assert "\n17\n" not in out


def test_sentence_splitter_protects_abbreviations():
    s = split_sentences("Mr. K. R. Sharma appeared for M/s ABC Ltd. in Crl. A. No. 5 of 2020. The appeal failed.")
    assert len(s) == 2


def test_chunker_respects_limit_and_loses_nothing():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(60))
    chunks = chunk_text(text, 50)
    assert all(len(c.split()) <= 50 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == len(text.split())


def test_eval_systems_return_text_and_citation_counts():
    from lawlens.summariser import evaluate as ev
    text = SAMPLE.read_text()
    for name in ("lead", "cues"):
        out, checked, verified = ev.make_system(name, None, 200, 0.35)(text)
        assert out.split() and verified == checked
    assert ev.make_system("cues", None, 200, 0.35)(text)[1] > 0     # cues system quotes the source
    assert ev.make_system("lead", None, 200, 0.35)(text)[1] == 0    # lead baseline has no citations


def _long_doc(copies=8):
    paras = [p.text for p in _paras() if p.kind == "para"]
    body = [t.split(". ", 1)[-1] if t[:2].strip(".").isdigit() else t for t in paras]
    lines = [f"{i + 1}. {t}" for i, t in enumerate(body * copies)]
    return split_paragraphs(clean_text("\n".join(lines)))


def test_scaled_caps_fill_long_budgets_and_leave_450_unchanged():
    long = _long_doc()
    at450_a = summarise(long, None, 450, scale_caps=True).n_words()
    at450_b = summarise(long, None, 450, scale_caps=False).n_words()
    assert at450_a == at450_b
    at900_scaled = summarise(long, None, 900, scale_caps=True).n_words()
    at900_fixed = summarise(long, None, 900, scale_caps=False).n_words()
    assert at900_scaled >= at900_fixed


def test_custom_weights_run_and_stay_verifiable():
    paras = _paras()
    s = summarise(paras, None, 300, weights=(0.6, 0.2, 0.2))
    v = verify_citations(s, paras)
    assert v["checked"] > 0 and v["verified"] == v["checked"]


def test_paired_bootstrap_detects_consistent_gain_and_zero_effect():
    from lawlens.summariser import evaluate as ev
    rng = np.random.default_rng(1)
    base = rng.normal(0.4, 0.1, 60)
    mean, lo, hi = ev.paired_bootstrap(base + 0.02, base)         # tiny but perfectly consistent gain
    assert lo > 0 and abs(mean - 0.02) < 1e-9
    mean, lo, hi = ev.paired_bootstrap(base + rng.normal(0, 0.05, 60), base)   # pure noise
    assert lo < 0 < hi


def test_unnumbered_text_falls_back_to_blocks():
    text = "\n\n".join(f"Paragraph {i} says something reasonably long about the matter at hand today." for i in range(8))
    paras = split_paragraphs(clean_text(text))
    assert len(paras) == 8 and paras[0].label == "P1"
