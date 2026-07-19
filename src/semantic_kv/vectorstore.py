"""Vector index over cache fragments.

This is where a Qdrant-style store earns its place in the design: the index
holds *cache fragments* (one point per cached chunk KV segment), not
documents. Lookups answer "do we already hold KV for something semantically
close to this chunk?"; payload filters implement span-level invalidation
(drop everything from doc X, or a specific chunk id).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np


@dataclass
class Point:
    id: str  # chunk cache id (= text hash)
    vector: np.ndarray
    payload: dict = field(default_factory=dict)


@dataclass
class Hit:
    id: str
    score: float
    payload: dict


class VectorStore(Protocol):
    def upsert(self, point: Point) -> None: ...

    def search(self, vector: np.ndarray, top_k: int = 1) -> list[Hit]: ...

    def delete(self, ids: Iterable[str] | None = None, doc_id: str | None = None) -> list[str]:
        """Delete by explicit ids and/or payload doc_id filter; return deleted ids."""
        ...

    def __len__(self) -> int: ...


class InMemoryVectorStore:
    """Brute-force cosine index. Fine into the tens of thousands of fragments."""

    def __init__(self):
        self._points: dict[str, Point] = {}

    def upsert(self, point: Point) -> None:
        self._points[point.id] = point

    def search(self, vector: np.ndarray, top_k: int = 1) -> list[Hit]:
        if not self._points:
            return []
        ids = list(self._points)
        mat = np.stack([self._points[i].vector for i in ids])
        scores = mat @ vector
        order = np.argsort(-scores)[:top_k]
        return [
            Hit(id=ids[i], score=float(scores[i]), payload=dict(self._points[ids[i]].payload))
            for i in order
        ]

    def delete(self, ids: Iterable[str] | None = None, doc_id: str | None = None) -> list[str]:
        doomed = set(ids or [])
        if doc_id is not None:
            doomed |= {p.id for p in self._points.values() if p.payload.get("doc_id") == doc_id}
        deleted = [i for i in doomed if i in self._points]
        for i in deleted:
            del self._points[i]
        return deleted

    def __len__(self) -> int:
        return len(self._points)


class QdrantVectorStore:
    """Qdrant-backed fragment index (requires the `qdrant` extra).

    Uses deterministic UUIDs derived from chunk ids so upserts are idempotent.
    """

    def __init__(self, dim: int, collection: str = "semantic_kv_fragments",
                 url: str | None = None, client=None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = client or QdrantClient(url=url) if (client or url) else QdrantClient(":memory:")
        self._collection = collection
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    @staticmethod
    def _uuid(chunk_id: str) -> str:
        import uuid

        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"semantic-kv:{chunk_id}"))

    def upsert(self, point: Point) -> None:
        from qdrant_client.models import PointStruct

        payload = dict(point.payload)
        payload["chunk_id"] = point.id
        self._client.upsert(
            self._collection,
            points=[PointStruct(id=self._uuid(point.id), vector=point.vector.tolist(), payload=payload)],
        )

    def search(self, vector: np.ndarray, top_k: int = 1) -> list[Hit]:
        res = self._client.query_points(self._collection, query=vector.tolist(), limit=top_k)
        return [
            Hit(id=p.payload["chunk_id"], score=float(p.score), payload=dict(p.payload))
            for p in res.points
        ]

    def delete(self, ids: Iterable[str] | None = None, doc_id: str | None = None) -> list[str]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList

        deleted: list[str] = []
        if ids:
            ids = list(ids)
            self._client.delete(self._collection, points_selector=PointIdsList(
                points=[self._uuid(i) for i in ids]))
            deleted.extend(ids)
        if doc_id is not None:
            flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            hits, _ = self._client.scroll(self._collection, scroll_filter=flt, limit=10_000)
            doc_ids = [h.payload["chunk_id"] for h in hits]
            if doc_ids:
                self._client.delete(self._collection, points_selector=PointIdsList(
                    points=[self._uuid(i) for i in doc_ids]))
            deleted.extend(doc_ids)
        return deleted

    def __len__(self) -> int:
        return self._client.count(self._collection).count
