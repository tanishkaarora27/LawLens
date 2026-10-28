"""Evaluation helpers for IN-Abs: loading, baselines, ROUGE, BERTScore, bootstrap CIs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ingest import load_text
from .pipeline import summarise, verify_citations


# --------------------------------------------------------------------------- data

def find_inabs_dirs(root: str | Path, split: str = "test") -> tuple[Path, Path]:
    """Locate the judgement/ and summary/ folders of IN-Abs inside the extracted Zenodo archive."""
    root = Path(root)
    found = []
    for d in root.rglob("*"):
        if d.is_dir() and d.name.lower() in ("judgement", "judgment"):
            summ = d.parent / "summary"
            key = str(d).lower().replace("_", "-").replace(" ", "-")
            if summ.is_dir() and "in-abs" in key and split in key:
                found.append((d, summ))
    if len(found) != 1:
        listing = "\n".join(f"  {a}" for a, _ in found) or "  (none)"
        raise FileNotFoundError(
            f"Expected exactly one IN-Abs '{split}' judgement/summary folder pair under {root}, "
            f"found {len(found)}:\n{listing}\nPass --judgement_dir and --summary_dir explicitly."
        )
    return found[0]


def load_pairs(judgement_dir: str | Path, summary_dir: str | Path, n: int | None = None, seed: int = 0):
    jd, sd = Path(judgement_dir), Path(summary_dir)
    names = sorted(p.name for p in jd.glob("*.txt") if (sd / p.name).exists())
    if not names:
        raise FileNotFoundError(f"No matching .txt files in {jd} and {sd}")
    if n is not None and n < len(names):
        rng = np.random.default_rng(seed)
        names = sorted(rng.choice(names, size=n, replace=False).tolist())
    return [
        (nm, (jd / nm).read_text(encoding="utf-8", errors="ignore"),
         (sd / nm).read_text(encoding="utf-8", errors="ignore"))
        for nm in names
    ]


# --------------------------------------------------------------------------- systems

def lead_baseline(text: str, max_words: int) -> str:
    """First max_words words of the judgment body (header excluded), the standard sanity baseline."""
    paras = [p for p in load_text(text) if p.kind == "para"]
    words = " ".join(p.text for p in paras).split()
    return " ".join(words[:max_words])


SYSTEMS = ("lead", "cues", "full", "abs")


def make_system(name: str, embedder, max_words: int, min_score: float, compressor=None,
                weights=None, scale_caps: bool = True):
    """Return fn(text) -> (summary_text, quotes_checked, quotes_verified).

    lead : first max_words words of the judgment (baseline)
    cues : cue phrases + position only (no model)
    full : cues + pretrained sentence-embedding similarity + position (extractive, verbatim quotes)
    abs  : `full` selection, then each cited paragraph rewritten by pretrained DistilBART (paraphrased)
    """
    if name not in SYSTEMS:
        raise ValueError(f"unknown system {name!r}; choose from {SYSTEMS}")
    if name == "lead":
        return lambda text: (lead_baseline(text, max_words), 0, 0)

    emb = embedder if name in ("full", "abs") else None
    comp = compressor if name == "abs" else None

    def run(text: str):
        paras = load_text(text)
        s = summarise(paras, emb, max_words, min_score, comp, weights, scale_caps)
        v = verify_citations(s, paras)
        return s.to_text(), v["checked"], v["verified"]

    return run


# --------------------------------------------------------------------------- metrics

def rouge_per_doc(candidates: list[str], references: list[str]) -> dict[str, np.ndarray]:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    out = {"rouge1": [], "rouge2": [], "rougeL": []}
    for c, r in zip(candidates, references):
        s = sc.score(r, c)
        for k in out:
            out[k].append(s[k].fmeasure)
    return {k: np.array(v) for k, v in out.items()}


def bertscore_per_doc(candidates, references, model_type="roberta-large", batch_size=16) -> np.ndarray:
    from bert_score import score

    _, _, f1 = score(candidates, references, lang="en", model_type=model_type,
                     batch_size=batch_size, verbose=False)
    return f1.numpy()


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean per-document difference (a - b) and its 95% bootstrap CI. Much more sensitive than
    comparing two separate CIs, because both systems are scored on the same documents."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% bootstrap confidence interval over documents."""
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
