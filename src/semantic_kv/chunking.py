"""Chunk identity and normalization.

A Chunk is the unit of caching: a span of retrieved context text plus the
identity of the document it came from. Cache keys are derived from a
normalized form of the text so that whitespace jitter does not defeat the
exact-match fast path, while the semantic path (embeddings) handles real
paraphrase-level drift.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field


_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class Chunk:
    """A retrieved context span."""

    text: str
    doc_id: str = ""
    metadata: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def hash(self) -> str:
        return text_hash(self.text)
