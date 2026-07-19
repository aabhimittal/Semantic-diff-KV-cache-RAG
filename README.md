# semantic-kv — Semantic-diff KV cache for RAG

**The standard RAG failure:** you re-encode near-identical context on every turn.
Retrieval returns 80%-overlapping chunks, and the model pays full prefill for all
of them, every time. Exact-match prefix caching (vLLM, SGLang radix caches) only
helps when the prefix is *byte-identical* — one reordered or lightly edited chunk
and the cache is cold.

`semantic-kv` is a prefix-caching layer keyed on **semantic similarity of
retrieved chunks**, not exact match. When a chunk of the new context is
semantically close to something already cached, its KV is reused (re-rotated to
its new position); only the delta — genuinely new chunks, plus an optional
repair budget — is computed.

```
turn 1:  [chunk A][chunk B][chunk C][query₁]   → full prefill, segments cached
turn 2:  [chunk A][chunk B'][chunk D][query₂]  → A reused, B' served from B
                                                  (cos 0.94), only D + query fresh
```

## The honest caveat (which is the whole project)

The KV of a shared span is **not** actually reusable when the preceding tokens
differ, for two reasons:

1. **Position** — under RoPE, keys are rotated by their absolute position.
   This part is *exactly* fixable: rotations compose, so keys cached at
   position `p` move to position `q` by applying one extra rotation `R(q−p)`
   ([`rope.py`](src/semantic_kv/rope.py)). No approximation.
2. **Cross-attention** — a cached chunk was encoded standalone, so its hidden
   states never attended to the chunks that now precede it. This is a real
   approximation and it does not go away.

So the project's core deliverable is **quantifying when the approximation
holds** ([`benchmarks/divergence.py`](benchmarks/divergence.py)), and a knob to
buy fidelity back: **blending** (`recompute_ratio`) recomputes the leading
fraction of each reused chunk against its true prefix, CacheBlend-style. At
`recompute_ratio=1.0` the pipeline is *numerically identical* to a vanilla
forward pass — the test suite asserts this to a 1e-4 tolerance
([`tests/test_equivalence.py`](tests/test_equivalence.py)) — which pins the
approximation to a single, tunable knob.

## Architecture

```
                       ┌────────────────────────────┐
 retrieved chunks ───▶ │  SemanticKVCache.lookup    │
                       │  1. exact  (text hash)     │
                       │  2. semantic (vector top-1 │
                       │     ≥ threshold)           │
                       │  3. miss                   │
                       └──────────┬─────────────────┘
        ┌─────────────────────────┼──────────────────────────┐
        ▼                         ▼                          ▼
┌───────────────┐        ┌────────────────┐         ┌───────────────┐
│ VectorStore   │        │ KVStore (LRU)  │         │ assemble()    │
│ fragments,    │        │ chunk_id →     │         │ reuse: RoPE   │
│ not documents │        │ KVSegment      │         │ re-rotate     │
│ (in-mem or    │        │ span-level     │         │ blend: repair │
│  Qdrant)      │        │ invalidation   │         │ miss: fresh   │
└───────────────┘        └────────────────┘         └───────────────┘
```

- **Vector store fronts the KV cache** — it indexes *cache fragments* (one
  point per cached chunk segment), not documents. This is where a Qdrant-style
  store earns its place: payload filters (`doc_id`) implement span-level
  invalidation, and nearest-fragment search implements the semantic key.
  In-memory brute-force store by default; `QdrantVectorStore` adapter included
  (`pip install "semantic-kv[qdrant]"`).
- **KV store** holds per-chunk segments (per-layer key/value tensors encoded
  standalone), LRU-bounded by bytes, kept in sync with the vector index on
  eviction and invalidation.
- **Semantic hits substitute text**: serving chunk B for near-duplicate B'
  means the model literally reads B's tokens. That is the semantic-cache bet —
  if cosine ≥ threshold, the informational content is interchangeable for the
  downstream answer. Tune `similarity_threshold` to taste (default 0.90).
- **Misses populate the cache**: a missed chunk is computed fresh in-context
  for this request *and* encoded standalone for future reuse.

## Quantifying the approximation

`benchmarks/divergence.py` compares cached-assembly prefills against exact
recomputation on identical token sequences, sweeping context overlap and blend
ratio. Metrics: mean KL divergence of next-token distributions over a fixed
continuation, top-1 agreement, and the fraction of prompt tokens actually
computed (1 − that = prefill saving).

```bash
python benchmarks/divergence.py --model Qwen/Qwen2.5-0.5B-Instruct
python benchmarks/divergence.py --tiny   # offline smoke run, random weights
```

Results land in [`results/`](results/) as JSON + a markdown table.

### Measured: Qwen2.5-0.5B-Instruct (CPU, fp32, ~340-token contexts)

| overlap | blend ratio | KL(exact‖cached) | top-1 agree | prefill saving |
|---|---|---|---|---|
| 100% | 0.00 | 0.923 | 58.8% | **78.2%** |
| 100% | 0.15 | 0.327 | 82.3% | **63.6%** |
| 100% | 0.30 | 0.227 | 88.2% | 52.7% |
| 100% | 1.00 | 0.000 | 100% | 0% |
| 80% | 0.00 | 0.940 | 47.1% | **64.3%** |
| 80% | 0.15 | 0.338 | 88.2% | **52.7%** |
| 80% | 0.30 | 0.244 | 88.2% | 43.8% |
| 50% | 0.15 | 0.299 | 100% | 35.7% |
| 50% | 0.30 | 0.241 | 94.1% | 29.6% |

The read on these numbers:

- **Pure reuse (ratio 0) is measurably lossy.** KL ≈ 0.9 and roughly half the
  next-token argmaxes flip. Position re-rotation alone does not make chunk KV
  interchangeable — the cross-attention deficit is real.
- **A small blend budget buys most of it back.** Recomputing just the leading
  15% of each reused chunk drops KL ~3× and lifts top-1 agreement to 82–100%,
  while still skipping half to two-thirds of prefill compute. This is the same
  regime CacheBlend reports (~15% recompute ≈ full-recompute quality).
- **KL falls monotonically with blend ratio and is exactly 0 at 1.0** — the
  approximation is confined to the knob, as the equivalence test guarantees.
- Wall-clock: at 100% overlap, cached prefill ran 293 ms vs 1165 ms exact
  (~4×) at ratio 0. Blended configs pay per-chunk forward-call overhead in
  this `transformers`-level prototype; a fused serving-engine implementation
  would recover most of that.

Where the fidelity/compute trade sits for *your* model and corpus is exactly
what the harness measures — rerun it with your checkpoint.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from semantic_kv import Chunk, SemanticKVCache, CachedRAGPipeline

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

cache = SemanticKVCache(
    model, tok,
    similarity_threshold=0.90,  # semantic-hit bar (cosine)
    recompute_ratio=0.15,       # blend budget: fraction of each reused chunk repaired
)
pipe = CachedRAGPipeline(cache)

chunks = [Chunk(text, doc_id="handbook") for text in retrieved_passages]
result = pipe.generate("What does the policy say about refunds?", chunks)
print(result.text)
print(result.stats.as_dict())   # reused/blended/fresh token counts, hit kinds
```

Span-level invalidation when a source document changes:

```python
cache.invalidate(doc_id="handbook")        # drop every fragment from that doc
cache.invalidate(chunk_id=some_chunk.hash) # or one span
```

Qdrant as the fragment index:

```python
from semantic_kv import QdrantVectorStore
cache = SemanticKVCache(model, tok, vector_store=QdrantVectorStore(dim=512, url="http://localhost:6333"))
```

## Scope and limits

- **RoPE models only** (Llama / Qwen / Mistral families). Learned-absolute-
  position models (GPT-2) can't re-rotate; ALiBi would need a different shift.
- Batch size 1 per assembly; segments are stored on the model's device/dtype.
- Chunk boundaries are token boundaries by construction — the exact-recompute
  baseline is defined over piecewise tokenization (see `tests/test_equivalence.py`).
- Standalone chunk encoding prepends BOS (dropped from the stored segment) so
  chunks are encoded in-distribution; this is part of the approximation.
- Sliding-window attention and cross-request batching are out of scope here;
  a production version would live inside the serving engine (vLLM/SGLang)
  rather than above `transformers`.

## Repo layout

```
src/semantic_kv/
  chunking.py      chunk identity, normalization, hashing
  embeddings.py    HashingEmbedder (deterministic) / sentence-transformers
  vectorstore.py   fragment index: in-memory + Qdrant adapter
  kv_store.py      LRU segment store, span-level invalidation
  rope.py          exact key re-rotation (position transplant)
  cache.py         lookup → assemble orchestrator (the core)
  pipeline.py      generation + continuation scoring on top of the cache
benchmarks/        divergence & savings study
tests/             offline suite (tiny random Llama), incl. exact-equivalence anchor
space/             Gradio demo for Hugging Face Spaces
docs/              GitHub Pages explainer
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests run offline against a randomly initialized 2-layer Llama, so CI needs no
model downloads and no GPU.

## References

- CacheBlend: fast LLM serving with cached knowledge fusion (Yao et al., 2024)
- PromptCache: modular attention reuse for low-latency inference (Gim et al., 2023)
- SGLang RadixAttention — exact-prefix reuse this project generalizes from
- StreamingLLM — attention sinks, why the first tokens' KV matters

## License

MIT — see [LICENSE](LICENSE).
