"""Evaluate Module 1 on IN-Abs (Indian Supreme Court judgments + headnote summaries).

Examples:
    python scripts/eval_inabs.py --root data/inabs --split test --max_words 450,900
    python scripts/eval_inabs.py --root data/inabs --split train --n 50 --seed 1 --weights 0.3,0,0.7 --systems lead,cues
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lawlens.summariser import evaluate as ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="folder where dataset.zip from Zenodo was extracted")
    ap.add_argument("--judgement_dir")
    ap.add_argument("--summary_dir")
    ap.add_argument("--split", default="test", help="test (100 docs) or train; tune on train, report on test")
    ap.add_argument("--n", type=int, default=None, help="random subset size (default: all)")
    ap.add_argument("--max_words", default="450",
                    help="summary length budget; comma list runs a length sweep, e.g. 300,450,900")
    ap.add_argument("--min_score", type=float, default=0.35)
    ap.add_argument("--systems", default="lead,cues,full", help="any of: lead,cues,full,abs")
    ap.add_argument("--weights", default=None,
                    help="w_cue,w_emb,w_pos, e.g. 0.3,0,0.7 (default 0.45,0.35,0.20)")
    ap.add_argument("--no_scale_caps", action="store_true",
                    help="use the original fixed per-role point caps")
    ap.add_argument("--bert_model", default="roberta-large", help="e.g. distilroberta-base for a faster run")
    ap.add_argument("--no_bertscore", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="seed for choosing the document subset")
    ap.add_argument("--out", default="results/module1_inabs")
    a = ap.parse_args()

    if a.judgement_dir and a.summary_dir:
        jd, sd = a.judgement_dir, a.summary_dir
    elif a.root:
        jd, sd = ev.find_inabs_dirs(a.root, a.split)
    else:
        ap.error("give --root, or both --judgement_dir and --summary_dir")

    pairs = ev.load_pairs(jd, sd, a.n, a.seed)
    names = [p[0] for p in pairs]
    docs = [p[1] for p in pairs]
    refs = [p[2] for p in pairs]
    gold_len = float(np.mean([len(r.split()) for r in refs]))
    print(f"{len(pairs)} documents from {jd}; mean gold summary = {gold_len:.0f} words")

    systems = [s.strip() for s in a.systems.split(",") if s.strip()]
    budgets = [int(x) for x in a.max_words.split(",")]
    weights = tuple(float(x) for x in a.weights.split(",")) if a.weights else None
    if weights is not None and len(weights) != 3:
        ap.error("--weights needs exactly three numbers: w_cue,w_emb,w_pos")
    scale_caps = not a.no_scale_caps

    embedder = compressor = None
    if "full" in systems or "abs" in systems:
        from lawlens.summariser.embedder import Embedder
        embedder = Embedder()
    if "abs" in systems:
        from lawlens.summariser.abstractive import AbstractiveCompressor
        compressor = AbstractiveCompressor()

    results, per_doc = {}, {}
    for budget in budgets:
        for sysname in systems:
            key = f"{sysname}@{budget}"
            fn = ev.make_system(sysname, embedder, budget, a.min_score, compressor, weights, scale_caps)
            t0 = time.time()
            outs = [fn(d) for d in docs]
            secs = time.time() - t0
            cands = [o[0] for o in outs]
            empty = sum(1 for c in cands if not c.strip())
            cands = [c if c.strip() else "empty" for c in cands]    # keep metric code from crashing
            checked = sum(o[1] for o in outs)
            verified = sum(o[2] for o in outs)

            r = ev.rouge_per_doc(cands, refs)
            docs_scores = {k: v.tolist() for k, v in r.items()}
            row = {k: ev.bootstrap_ci(v) for k, v in r.items()}
            if not a.no_bertscore:
                b = ev.bertscore_per_doc(cands, refs, a.bert_model)
                docs_scores["bertscore_f1"] = b.tolist()
                row["bertscore_f1"] = ev.bootstrap_ci(b)
            row["mean_words"] = float(np.mean([len(c.split()) for c in cands]))
            row["empty_outputs"] = empty
            row["quotes_verified"] = f"{verified}/{checked}" if checked else "n/a"
            row["seconds"] = round(secs, 1)
            results[key] = row
            per_doc[key] = docs_scores
            print(key, {k: (round(v[0], 4) if isinstance(v, tuple) else v) for k, v in row.items()})

    metrics = [c for c in ["rouge1", "rouge2", "rougeL", "bertscore_f1"] if c in next(iter(per_doc.values()))]

    # paired difference against the lead baseline at the same budget
    deltas = {}
    for key in results:
        sysname, budget = key.split("@")
        if sysname == "lead" or f"lead@{budget}" not in per_doc:
            continue
        deltas[key] = {m: ev.paired_bootstrap(per_doc[key][m], per_doc[f"lead@{budget}"][m]) for m in metrics}

    meta = {"n_docs": len(pairs), "split": a.split, "budgets": budgets, "min_score": a.min_score,
            "bert_model": None if a.no_bertscore else a.bert_model, "gold_mean_words": gold_len,
            "weights": list(weights) if weights else "default", "scale_caps": scale_caps,
            "seed": a.seed, "docs": names}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(
        {"meta": meta, "results": results, "paired_vs_lead": deltas, "per_doc": per_doc}, indent=2))

    lines = [f"IN-Abs {a.split} split, n={len(pairs)} (gold summaries average {gold_len:.0f} words). "
             f"Weights (cue,emb,pos): {meta['weights']}. Mean F1 [95% bootstrap CI over documents].", "",
             "| System @ word budget | " + " | ".join(metrics) + " | words | quotes verified |",
             "|---|" + "---|" * (len(metrics) + 2)]
    for key, row in results.items():
        cells = [f"{row[c][0]:.3f} [{row[c][1]:.3f}, {row[c][2]:.3f}]" for c in metrics]
        lines.append(f"| {key} | " + " | ".join(cells) + f" | {row['mean_words']:.0f} | {row['quotes_verified']} |")
    if deltas:
        lines += ["", "Paired difference vs the lead baseline at the same budget (system minus lead, 95% bootstrap CI). "
                  "`*` marks a CI that excludes 0.", "",
                  "| System | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
        for key, d in deltas.items():
            cells = []
            for m in metrics:
                mean, lo, hi = d[m]
                star = "*" if (lo > 0 or hi < 0) else ""
                cells.append(f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved {out.with_suffix('.json')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
