"""Summarise one judgment from the command line (quick check without the Streamlit UI).

    python scripts/summarise_cli.py samples/sample_judgment.txt
    python scripts/summarise_cli.py my_judgment.pdf --max_words 600 --no_embeddings
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lawlens.summariser import load_document, summarise, verify_citations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max_words", type=int, default=450)
    ap.add_argument("--no_embeddings", action="store_true")
    a = ap.parse_args()

    p = Path(a.path)
    paras = load_document(p.name, p.read_bytes())
    embedder = None
    if not a.no_embeddings:
        from lawlens.summariser.embedder import Embedder
        embedder = Embedder()

    summary = summarise(paras, embedder, a.max_words)
    print(summary.to_markdown())
    v = verify_citations(summary, paras)
    print(f"Citation check: {v['verified']}/{v['checked']} quoted sentences found verbatim in the cited paragraph")
    print(f"Document: {len(paras)} paragraphs; summary: {summary.n_words()} words")


if __name__ == "__main__":
    main()
