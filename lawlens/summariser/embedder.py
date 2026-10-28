"""Thin wrapper around a pretrained sentence encoder (no fine-tuning)."""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # imported lazily so tests need no torch

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)
