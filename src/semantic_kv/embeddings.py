"""Pluggable text embedders for semantic cache lookup.

The cache only needs *relative* similarity between near-duplicate chunks, not
state-of-the-art retrieval quality, so a cheap deterministic embedder is the
default. A sentence-transformers backend is available as an extra.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray:
        """Return an L2-normalized vector of shape (dim,)."""
        ...


_TOKEN = re.compile(r"\w+")


class HashingEmbedder:
    """Deterministic bag-of-ngrams hashing embedder.

    Word unigrams/bigrams plus character trigrams are hashed into a fixed-size
    signed feature vector. Near-identical texts land at cosine ~1, edits move
    them smoothly away — exactly the behavior the cache lookup needs, with no
    model download and no nondeterminism.
    """

    def __init__(self, dim: int = 512, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def _slot(self, feature: str) -> tuple[int, float]:
        h = hashlib.blake2b(
            feature.encode("utf-8"), digest_size=8, salt=str(self.seed).encode()
        ).digest()
        v = int.from_bytes(h, "little")
        return v % self.dim, 1.0 if (v >> 63) & 1 else -1.0

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = [w.lower() for w in _TOKEN.findall(text)]
        feats: list[str] = []
        feats.extend(f"w:{w}" for w in words)
        feats.extend(f"b:{a}_{b}" for a, b in zip(words, words[1:]))
        joined = " ".join(words)
        feats.extend(f"c:{joined[i:i + 3]}" for i in range(max(0, len(joined) - 2)))
        for f in feats:
            idx, sign = self._slot(f)
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


class SentenceTransformerEmbedder:
    """sentence-transformers backend (requires the `embeddings` extra)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)
