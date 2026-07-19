"""Storage for per-chunk KV segments with LRU eviction and span-level invalidation."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch


@dataclass
class KVSegment:
    """KV tensors for one chunk, encoded standalone.

    keys/values: lists (one entry per layer) of (1, kv_heads, chunk_len, head_dim)
    tensors. `base_position` records the absolute position at which token 0 of
    the segment was encoded (1 when a BOS token was prepended and dropped), so
    reuse at offset q applies a single re-rotation by q - base_position.
    """

    chunk_id: str
    doc_id: str
    text: str
    token_ids: list[int]
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    base_position: int
    created_at: float = field(default_factory=time.time)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def nbytes(self) -> int:
        return sum(t.element_size() * t.nelement() for t in self.keys) * 2


class KVStore:
    """LRU-bounded map chunk_id -> KVSegment."""

    def __init__(self, max_bytes: int = 2 * 1024**3):
        self.max_bytes = max_bytes
        self._segments: OrderedDict[str, KVSegment] = OrderedDict()
        self._bytes = 0

    def put(self, seg: KVSegment) -> list[str]:
        """Insert a segment; returns ids evicted to stay under max_bytes."""
        if seg.chunk_id in self._segments:
            self._bytes -= self._segments.pop(seg.chunk_id).nbytes
        self._segments[seg.chunk_id] = seg
        self._bytes += seg.nbytes
        evicted = []
        while self._bytes > self.max_bytes and len(self._segments) > 1:
            old_id, old = self._segments.popitem(last=False)
            self._bytes -= old.nbytes
            evicted.append(old_id)
        return evicted

    def get(self, chunk_id: str) -> KVSegment | None:
        seg = self._segments.get(chunk_id)
        if seg is not None:
            self._segments.move_to_end(chunk_id)
        return seg

    def delete(self, chunk_id: str) -> bool:
        seg = self._segments.pop(chunk_id, None)
        if seg is None:
            return False
        self._bytes -= seg.nbytes
        return True

    def ids_for_doc(self, doc_id: str) -> list[str]:
        return [cid for cid, s in self._segments.items() if s.doc_id == doc_id]

    @property
    def total_bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._segments)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._segments
