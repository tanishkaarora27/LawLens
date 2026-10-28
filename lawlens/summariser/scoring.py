"""Zero-shot role scores for every paragraph: cue phrases + embedding similarity + position."""

from __future__ import annotations

import numpy as np

from .roles import PROTOTYPES, ROLES, cue_score, position_prior
from .textutils import Paragraph, chunk_text


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-9)


def score_paragraphs(
    paras: list[Paragraph],
    embedder=None,
    w_cue: float = 0.45,
    w_emb: float = 0.35,
    w_pos: float = 0.20,
) -> dict[str, np.ndarray]:
    """Return {role: scores in [0, 1]} aligned with `paras`.

    With embedder=None the embedding term is dropped and the other weights are renormalised
    (the "cues + position only" ablation used in the evaluation).
    """
    n = len(paras)
    rel = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)

    emb_sim: dict[str, np.ndarray] | None = None
    if embedder is not None and n:
        chunks, owner = [], []
        for i, p in enumerate(paras):
            for c in chunk_text(p.text, 120):
                chunks.append(c)
                owner.append(i)
        owner = np.asarray(owner)
        E = embedder.encode(chunks)
        emb_sim = {}
        for role in ROLES:
            proto = embedder.encode(PROTOTYPES[role])
            sims = (E @ proto.T).max(axis=1)            # best prototype per chunk
            per_para = np.full(n, -1.0)
            np.maximum.at(per_para, owner, sims)         # best chunk per paragraph
            emb_sim[role] = _minmax(per_para)

    scores = {}
    for role in ROLES:
        cue = np.array([cue_score(p.text, role) for p in paras])
        pos = np.array([position_prior(role, r) for r in rel])
        if emb_sim is not None:
            scores[role] = w_cue * cue + w_emb * emb_sim[role] + w_pos * pos
        else:
            scores[role] = (w_cue * cue + w_pos * pos) / (w_cue + w_pos)
    return scores
