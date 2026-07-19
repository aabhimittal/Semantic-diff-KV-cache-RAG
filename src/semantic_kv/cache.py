"""SemanticKVCache: prefix caching keyed on semantic similarity of chunks.

Lookup order per chunk:
  1. exact — normalized-text hash match (free, lossless w.r.t. the chunk text)
  2. semantic — nearest fragment in the vector index with cosine >= threshold;
     the *cached* chunk's tokens and KV stand in for the near-duplicate
  3. miss — the chunk is computed fresh in context, and also encoded
     standalone and admitted to the cache for future requests

Reused segments are placed at their new absolute position by RoPE
re-rotation (exact for position; see rope.py) and, optionally, the leading
`recompute_ratio` fraction of their tokens is recomputed against the true
prefix ("blending") to repair cross-chunk attention. recompute_ratio=1.0
degenerates to exact full recomputation — tests rely on that equivalence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from .chunking import Chunk
from .embeddings import Embedder, HashingEmbedder
from .kv_store import KVSegment, KVStore
from .metrics import AssemblyStats
from .rope import get_inv_freq, rerotate_keys
from .vectorstore import InMemoryVectorStore, Point, VectorStore


@dataclass
class Assembled:
    cache: DynamicCache
    input_ids: list[int]          # tokens the model has effectively "read"
    last_logits: torch.Tensor     # logits after the final assembled token, (vocab,)
    stats: AssemblyStats


class SemanticKVCache:
    def __init__(
        self,
        model,
        tokenizer,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        similarity_threshold: float = 0.90,
        recompute_ratio: float = 0.0,
        max_bytes: int = 2 * 1024**3,
        chunk_bos: bool = True,
        store_on_miss: bool = True,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.embedder = embedder or HashingEmbedder()
        self.vectors = vector_store or InMemoryVectorStore()
        self.kv = KVStore(max_bytes=max_bytes)
        self.similarity_threshold = similarity_threshold
        self.recompute_ratio = recompute_ratio
        self.chunk_bos = chunk_bos
        self.store_on_miss = store_on_miss
        self.inv_freq = get_inv_freq(model)
        self.device = next(model.parameters()).device

    # ---------------------------------------------------------------- lookup

    def lookup(self, chunk: Chunk) -> tuple[KVSegment | None, float, str]:
        """Return (segment, score, kind) with kind in {exact, semantic, miss}."""
        seg = self.kv.get(chunk.hash)
        if seg is not None:
            return seg, 1.0, "exact"
        hits = self.vectors.search(self.embedder.embed(chunk.text), top_k=1)
        if hits and hits[0].score >= self.similarity_threshold:
            seg = self.kv.get(hits[0].id)
            if seg is not None:
                return seg, hits[0].score, "semantic"
        return None, 0.0, "miss"

    # ---------------------------------------------------------------- encode

    def _tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def encode_chunk(self, chunk: Chunk) -> KVSegment:
        """Encode a chunk standalone and admit its KV segment to the cache."""
        token_ids = self._tokenize(chunk.text)
        bos = self.tokenizer.bos_token_id if self.chunk_bos else None
        ids = ([bos] if bos is not None else []) + token_ids
        skip = 1 if bos is not None else 0
        input_ids = torch.tensor([ids], device=self.device)
        out = self.model(input_ids, use_cache=True)
        layers = out.past_key_values.layers
        seg = KVSegment(
            chunk_id=chunk.hash,
            doc_id=chunk.doc_id,
            text=chunk.text,
            token_ids=token_ids,
            keys=[l.keys[:, :, skip:, :].detach().clone() for l in layers],
            values=[l.values[:, :, skip:, :].detach().clone() for l in layers],
            base_position=skip,
        )
        evicted = self.kv.put(seg)
        if evicted:
            self.vectors.delete(ids=evicted)
        self.vectors.upsert(Point(
            id=seg.chunk_id,
            vector=self.embedder.embed(chunk.text),
            payload={"doc_id": chunk.doc_id, "num_tokens": seg.num_tokens},
        ))
        return seg

    def warm(self, chunks: list[Chunk]) -> int:
        """Pre-populate the cache; returns number of segments encoded."""
        n = 0
        for c in chunks:
            if c.hash not in self.kv:
                self.encode_chunk(c)
                n += 1
        return n

    # ------------------------------------------------------------ invalidate

    def invalidate(self, doc_id: str | None = None, chunk_id: str | None = None) -> int:
        """Span-level invalidation: drop fragments by document or chunk identity."""
        ids: list[str] = []
        if chunk_id is not None:
            ids.append(chunk_id)
        if doc_id is not None:
            ids.extend(self.kv.ids_for_doc(doc_id))
        removed = 0
        for cid in set(ids):
            removed += self.kv.delete(cid)
        self.vectors.delete(ids=set(ids), doc_id=doc_id)
        return removed

    # -------------------------------------------------------------- assembly

    @torch.no_grad()
    def _forward_into(self, cache: DynamicCache, token_ids: list[int], pos: int) -> torch.Tensor:
        """Run tokens through the model appending to `cache`; return last-token logits."""
        ids = torch.tensor([token_ids], device=self.device)
        n = len(token_ids)
        out = self.model(
            ids,
            past_key_values=cache,
            use_cache=True,
            attention_mask=torch.ones(1, pos + n, dtype=torch.long, device=self.device),
            position_ids=torch.arange(pos, pos + n, device=self.device)[None],
            cache_position=torch.arange(pos, pos + n, device=self.device),
        )
        return out.logits[0, -1]

    def _append_segment_kv(self, cache: DynamicCache, seg: KVSegment,
                           pos: int, from_token: int) -> None:
        """Append seg.keys/values[from_token:] to the cache, re-rotated to `pos + from_token`."""
        delta = (pos + from_token) - (seg.base_position + from_token)
        for layer_idx, (k, v) in enumerate(zip(seg.keys, seg.values)):
            k = k[:, :, from_token:, :]
            v = v[:, :, from_token:, :]
            k = rerotate_keys(k.to(self.device), delta, self.inv_freq)
            cache.update(k, v.to(self.device), layer_idx)

    @torch.no_grad()
    def assemble(
        self,
        chunks: list[Chunk],
        prefix_text: str = "",
        suffix_text: str = "",
        recompute_ratio: float | None = None,
    ) -> Assembled:
        """Build the KV for `prefix + chunks + suffix`, reusing cached segments.

        prefix/suffix (instruction header, question) are always computed fresh.
        Returns the populated cache, the effective token ids, and the logits
        after the last token — ready for generation.
        """
        ratio = self.recompute_ratio if recompute_ratio is None else recompute_ratio
        cache = DynamicCache()
        stats = AssemblyStats()
        all_ids: list[int] = []
        last_logits: torch.Tensor | None = None
        pos = 0

        def run_fresh(token_ids: list[int]) -> None:
            nonlocal pos, last_logits
            if not token_ids:
                return
            last_logits = self._forward_into(cache, token_ids, pos)
            pos += len(token_ids)
            all_ids.extend(token_ids)

        bos = self.tokenizer.bos_token_id
        head = ([bos] if bos is not None else []) + self._tokenize(prefix_text)
        run_fresh(head)
        stats.fresh_tokens += len(head)

        for chunk in chunks:
            seg, score, kind = self.lookup(chunk)
            if seg is None:
                if self.store_on_miss:
                    self.encode_chunk(chunk)
                token_ids = self._tokenize(chunk.text)
                run_fresh(token_ids)
                stats.misses += 1
                stats.fresh_tokens += len(token_ids)
                continue

            if kind == "exact":
                stats.exact_hits += 1
            else:
                stats.semantic_hits += 1
                stats.substitutions.append(
                    {"requested": chunk.text[:80], "served": seg.text[:80], "score": round(score, 4)}
                )
            n_blend = min(seg.num_tokens, math.ceil(ratio * seg.num_tokens))
            if n_blend:
                last_logits = self._forward_into(cache, seg.token_ids[:n_blend], pos)
            if n_blend < seg.num_tokens:
                self._append_segment_kv(cache, seg, pos, from_token=n_blend)
            pos += seg.num_tokens
            all_ids.extend(seg.token_ids)
            stats.blended_tokens += n_blend
            stats.reused_tokens += seg.num_tokens - n_blend

        tail = self._tokenize(suffix_text)
        run_fresh(tail)
        stats.fresh_tokens += len(tail)

        if last_logits is None:
            # Degenerate case: everything reused, nothing fresh — poke the model
            # with the final cached token re-fed? Simpler: require some fresh
            # tail; callers always pass a suffix in practice.
            raise ValueError("assemble() needs at least one fresh token (prefix, suffix, or BOS)")

        stats.prompt_tokens = pos
        return Assembled(cache=cache, input_ids=all_ids, last_logits=last_logits, stats=stats)
