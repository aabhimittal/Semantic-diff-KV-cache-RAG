"""The design's anchor property: recompute_ratio=1.0 must be numerically
equivalent to a vanilla contiguous forward pass. This validates the whole
assembly machinery — token layout, position ids, cache plumbing — leaving
recompute_ratio<1.0 as the *only* source of approximation."""

import torch

from semantic_kv import Chunk, CachedRAGPipeline, SemanticKVCache

CHUNKS = [
    Chunk("Rivers deposit sediment where their current slows near the delta mouth.", doc_id="d1"),
    Chunk("Glacial melt contributes seasonal variation in downstream water volume.", doc_id="d2"),
    Chunk("Sediment cores record centuries of flood frequency in layered silt.", doc_id="d3"),
]
PREFIX = "Use the context to answer.\n\n"
SUFFIX = "\n\nQuestion: what do sediment cores record?\nAnswer:"


def vanilla_logits(model, tokenizer, device="cpu"):
    # Piecewise tokenization: chunk boundaries are token boundaries by
    # construction in a chunk-level cache, so the exact-recompute baseline
    # tokenizes each piece separately and runs one contiguous forward.
    ids: list[int] = []
    bos = tokenizer.bos_token_id
    if bos is not None:
        ids.append(bos)
    for piece in [PREFIX, *[c.text for c in CHUNKS], SUFFIX]:
        ids.extend(tokenizer(piece, add_special_tokens=False)["input_ids"])
    with torch.no_grad():
        out = model(torch.tensor([ids], device=device))
    return ids, out.logits[0, -1]


def test_full_blend_equals_vanilla_forward(model, tokenizer):
    cache = SemanticKVCache(model, tokenizer, recompute_ratio=1.0)
    cache.warm(CHUNKS)
    asm = cache.assemble(CHUNKS, prefix_text=PREFIX, suffix_text=SUFFIX)
    ref_ids, ref_logits = vanilla_logits(model, tokenizer)
    assert asm.input_ids == ref_ids
    assert torch.allclose(asm.last_logits, ref_logits, atol=1e-4), (
        (asm.last_logits - ref_logits).abs().max()
    )


def test_pure_reuse_is_close_but_runs(model, tokenizer):
    """ratio=0 is approximate by construction — assert it runs and produces
    finite logits over the full vocab, and that everything was reused."""
    cache = SemanticKVCache(model, tokenizer, recompute_ratio=0.0)
    cache.warm(CHUNKS)
    asm = cache.assemble(CHUNKS, prefix_text=PREFIX, suffix_text=SUFFIX)
    assert asm.stats.exact_hits == len(CHUNKS)
    assert asm.stats.blended_tokens == 0
    assert torch.isfinite(asm.last_logits).all()
    _, ref_logits = vanilla_logits(model, tokenizer)
    assert asm.last_logits.shape == ref_logits.shape


def test_partial_blend_reduces_divergence(model, tokenizer):
    """More blending should not increase divergence from the exact forward."""
    _, ref = vanilla_logits(model, tokenizer)
    ref_lp = torch.log_softmax(ref, dim=-1)

    def kl(ratio):
        cache = SemanticKVCache(model, tokenizer)
        cache.warm(CHUNKS)
        asm = cache.assemble(CHUNKS, prefix_text=PREFIX, suffix_text=SUFFIX,
                             recompute_ratio=ratio)
        lp = torch.log_softmax(asm.last_logits, dim=-1)
        return torch.nn.functional.kl_div(lp, ref_lp, log_target=True,
                                          reduction="sum").item()

    assert kl(1.0) <= kl(0.0) + 1e-6
    assert kl(1.0) < 1e-6


def test_generation_runs_end_to_end(model, tokenizer):
    cache = SemanticKVCache(model, tokenizer)
    cache.warm(CHUNKS)
    pipe = CachedRAGPipeline(cache)
    res = pipe.generate("what do sediment cores record?", CHUNKS, max_new_tokens=8)
    assert len(res.token_ids) <= 8
    assert res.stats.exact_hits == len(CHUNKS)


def test_score_continuation_shape(model, tokenizer):
    cache = SemanticKVCache(model, tokenizer)
    cache.warm(CHUNKS)
    pipe = CachedRAGPipeline(cache)
    cont = tokenizer("layered silt records floods", add_special_tokens=False)["input_ids"]
    logits = pipe.score_continuation("q?", CHUNKS, cont)
    assert logits.shape[0] == len(cont)
