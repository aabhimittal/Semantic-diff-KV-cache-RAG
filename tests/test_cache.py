import torch

from semantic_kv import Chunk, SemanticKVCache

DOC_A = "The Amazon rainforest produces roughly twenty percent of the world's oxygen supply."
DOC_B = "Photosynthesis in ocean plankton contributes about half of atmospheric oxygen renewal."
DOC_C = "Deforestation rates in the Amazon basin rose sharply during the last decade of study."


def make_cache(model, tokenizer, **kw):
    return SemanticKVCache(model, tokenizer, similarity_threshold=0.85, **kw)


def test_exact_hit_after_warm(model, tokenizer):
    cache = make_cache(model, tokenizer)
    chunk = Chunk(DOC_A, doc_id="a")
    cache.warm([chunk])
    seg, score, kind = cache.lookup(chunk)
    assert kind == "exact" and score == 1.0 and seg.num_tokens > 0


def test_semantic_hit_on_whitespace_and_small_edit(model, tokenizer):
    cache = make_cache(model, tokenizer)
    cache.warm([Chunk(DOC_A, doc_id="a")])
    # Same words, one word changed -> high hashing-embedder cosine, not exact.
    variant = DOC_A.replace("twenty", "20")
    seg, score, kind = cache.lookup(Chunk(variant, doc_id="a"))
    assert kind == "semantic"
    assert 0.85 <= score < 1.0
    assert seg.text == DOC_A  # cached text is served in place of the variant


def test_unrelated_text_misses(model, tokenizer):
    cache = make_cache(model, tokenizer)
    cache.warm([Chunk(DOC_A, doc_id="a")])
    seg, _, kind = cache.lookup(Chunk(DOC_B, doc_id="b"))
    assert kind == "miss" and seg is None


def test_span_level_invalidation(model, tokenizer):
    cache = make_cache(model, tokenizer)
    cache.warm([Chunk(DOC_A, doc_id="a"), Chunk(DOC_C, doc_id="a"), Chunk(DOC_B, doc_id="b")])
    assert len(cache.kv) == 3
    removed = cache.invalidate(doc_id="a")
    assert removed == 2
    assert len(cache.kv) == 1
    assert cache.lookup(Chunk(DOC_A, doc_id="a"))[2] == "miss"
    assert cache.lookup(Chunk(DOC_B, doc_id="b"))[2] == "exact"


def test_lru_eviction_keeps_vector_index_in_sync(model, tokenizer):
    cache = make_cache(model, tokenizer)
    one = cache.encode_chunk(Chunk(DOC_A, doc_id="a"))
    cache.kv.max_bytes = one.nbytes + 1  # room for ~one segment
    cache.encode_chunk(Chunk(DOC_B, doc_id="b"))
    assert len(cache.kv) == 1
    assert len(cache.vectors) == 1
    assert cache.lookup(Chunk(DOC_A, doc_id="a"))[2] == "miss"


def test_assemble_stats_and_reuse(model, tokenizer):
    cache = make_cache(model, tokenizer)
    chunks = [Chunk(DOC_A, doc_id="a"), Chunk(DOC_B, doc_id="b")]
    cache.warm(chunks)
    asm = cache.assemble(chunks, prefix_text="Context:", suffix_text="Question: why?")
    assert asm.stats.exact_hits == 2
    assert asm.stats.reused_tokens > 0
    assert asm.stats.misses == 0
    assert asm.stats.prompt_tokens == len(asm.input_ids)
    assert asm.last_logits.ndim == 1


def test_miss_populates_cache_for_next_request(model, tokenizer):
    cache = make_cache(model, tokenizer)
    chunks = [Chunk(DOC_A, doc_id="a")]
    first = cache.assemble(chunks, suffix_text="Q")
    assert first.stats.misses == 1
    second = cache.assemble(chunks, suffix_text="Q")
    assert second.stats.exact_hits == 1 and second.stats.misses == 0
    assert second.stats.reused_tokens > 0
