"""Structured, paragraph-cited summary of a judgment.

Pipeline:  paragraphs -> role scores (cues + embeddings + position) -> per-role selection under a
word budget -> best sentences inside each chosen paragraph -> points that each carry a citation.

Extractive by default, so every point is a verbatim quote and `verify_citations` can prove it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .roles import PROTOTYPES, ROLES, ROLE_TITLES, cue_score
from .scoring import score_paragraphs
from .textutils import Paragraph, normalize_ws, split_sentences

ROLE_ORDER = ["decision", "issues", "arguments", "facts", "reasoning"]   # most specific first
SHARES = {"facts": 0.20, "issues": 0.10, "arguments": 0.20, "reasoning": 0.35, "decision": 0.15}
MAX_POINTS = {"facts": 6, "issues": 4, "arguments": 6, "reasoning": 6, "decision": 2}
MAX_SENT_WORDS = 70

# Frozen after tuning on IN-Abs *train* (50 docs), confirmed on 50 other train docs, then reported
# once on the test split. (w_cue, w_emb, w_pos). Used whenever no embedding model is supplied.
VALIDATED_WEIGHTS = (0.3, 0.0, 0.7)


@dataclass
class Point:
    role: str
    label: str            # paragraph label to cite, e.g. "14"
    para_idx: int
    text: str             # what is shown
    sentences: list[str]  # source sentences the point is built from
    verbatim: bool        # True if `text` is made only of quoted sentences
    score: float
    cue_hit: bool = True  # False = chosen mostly by position; no legal cue phrase for this role (low confidence)


@dataclass
class Summary:
    points: list[Point]
    n_paragraphs: int
    settings: dict = field(default_factory=dict)

    def by_role(self) -> dict[str, list[Point]]:
        return {r: [p for p in self.points if p.role == r] for r in ROLES}

    def to_text(self) -> str:
        """Plain text with no headings or citation marks: what ROUGE / BERTScore are computed on."""
        return " ".join(p.text for r in ROLES for p in self.by_role()[r])

    def n_words(self) -> int:
        return len(self.to_text().split())

    def to_markdown(self) -> str:
        out = []
        for r in ROLES:
            out.append(f"## {ROLE_TITLES[r]}")
            pts = self.by_role()[r]
            if not pts:
                out.append("_Not clearly identified in this document._")
            for p in pts:
                tag = "" if p.verbatim else " (paraphrased)"
                tag += "" if p.cue_hit else " (low confidence: chosen by position only)"
                out.append(f"- {p.text} **[¶{p.label}]**{tag}")
            out.append("")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {"n_paragraphs": self.n_paragraphs, "settings": self.settings,
                "points": [asdict(p) for p in self.points]}


def _trim(sentence: str, max_words: int = MAX_SENT_WORDS) -> str:
    w = sentence.split()
    return sentence if len(w) <= max_words else " ".join(w[:max_words]) + " …"


def _pick_sentences(p: Paragraph, role: str, embedder, protos, word_budget: float):
    """Choose up to two sentences of a paragraph that best fit the role. Returns (texts, indices)."""
    sents = split_sentences(p.text)
    if not sents:
        return [p.text], [0]
    if p.n_words <= 45 and p.n_words <= word_budget:
        return [p.text], [0]
    sc = np.array([cue_score(s, role) for s in sents], dtype=float)
    if embedder is not None:
        sc = sc + (embedder.encode(sents) @ protos[role].T).max(axis=1)
    chosen, used = [], 0
    for i in np.argsort(-sc):
        w = min(len(sents[i].split()), MAX_SENT_WORDS)
        if chosen and (used + w > word_budget or len(chosen) >= 2):
            break
        chosen.append(int(i))
        used += w
    chosen.sort()
    return [_trim(sents[i]) for i in chosen], chosen


def summarise(
    paras: list[Paragraph],
    embedder=None,
    max_words: int = 450,
    min_score: float = 0.35,
    compressor=None,
    weights: tuple[float, float, float] | None = None,
    scale_caps: bool = True,
) -> Summary:
    """weights = (w_cue, w_emb, w_pos).

    None means: the validated cues+position weights when there is no embedding model (the
    configuration evaluated in the README), or the untuned defaults in scoring.py when an
    embedding model is supplied (experimental: it showed no measured gain).

    scale_caps=True lets the per-role point caps grow in proportion to max_words/450, so long
    budgets can actually be filled. False reproduces the original fixed caps.
    """
    if weights is None and embedder is None:
        weights = VALIDATED_WEIGHTS
    cidx = [i for i, p in enumerate(paras) if p.kind == "para" and p.n_words >= 8]
    cpar = [paras[i] for i in cidx]
    n = len(cpar)
    cap_scale = max(1.0, max_words / 450) if scale_caps else 1.0
    settings = {"max_words": max_words, "min_score": min_score,
                "embeddings": embedder is not None, "abstractive": compressor is not None,
                "weights": list(weights) if weights else "default", "scale_caps": scale_caps}
    if n == 0:
        return Summary([], len(paras), settings)

    scores = score_paragraphs(cpar, embedder, *weights) if weights else score_paragraphs(cpar, embedder)
    protos = {r: embedder.encode(PROTOTYPES[r]) for r in ROLES} if embedder is not None else None
    tail_start = n - max(3, math.ceil(0.15 * n))          # the outcome sits in the last paragraphs

    used: set[int] = set()
    points: list[Point] = []
    for role in ROLE_ORDER:
        budget, spent, k = SHARES[role] * max_words, 0, 0
        cap = math.ceil(MAX_POINTS[role] * cap_scale)
        for j in np.argsort(-scores[role]):
            s = float(scores[role][j])
            if s < min_score:
                break
            if j in used or (role == "decision" and j < tail_start):
                continue
            # an eligible paragraph needs at least one cue hit, or a clearly high combined score
            if cue_score(cpar[j].text, role) == 0 and s < 0.55:
                continue
            remaining = budget - spent
            if k >= cap or (k > 0 and remaining < 15):
                break
            texts, idxs = _pick_sentences(cpar[j], role, embedder, protos, max(remaining, 25))
            joined = ""
            for t, i in zip(texts, idxs):
                sep = "" if not joined else (" " if i == prev + 1 else " … ")
                joined += sep + t
                prev = i
            points.append(Point(role, cpar[j].label, cpar[j].idx, joined, texts, True, s,
                                cue_score(cpar[j].text, role) > 0))
            used.add(int(j))
            spent += len(joined.split())
            k += 1

    points.sort(key=lambda p: (ROLES.index(p.role), p.para_idx))

    if compressor is not None:
        by_idx = {p.idx: p for p in paras}
        for p in points:
            p.text = compressor.rewrite(by_idx[p.para_idx].text)
            p.verbatim = False

    return Summary(points, len(paras), settings)


def verify_citations(summary: Summary, paras: list[Paragraph]) -> dict:
    """Machine check: every quoted sentence must occur verbatim inside the paragraph it cites."""
    by_idx = {p.idx: normalize_ws(p.text) for p in paras}
    checked = ok = 0
    failures = []
    for pt in summary.points:
        if not pt.verbatim:
            continue
        for s in pt.sentences:
            checked += 1
            piece = normalize_ws(s[:-1].rstrip() if s.endswith("…") else s)
            if piece and piece in by_idx.get(pt.para_idx, ""):
                ok += 1
            else:
                failures.append({"label": pt.label, "sentence": s})
    return {"checked": checked, "verified": ok, "failures": failures}
