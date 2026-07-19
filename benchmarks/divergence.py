"""Quantify when semantic KV reuse holds up.

The KV of a shared span is not exactly reusable when the preceding tokens
differ (positional + cross-attention dependencies). Re-rotation fixes the
positional part exactly; this benchmark measures what remains, as a function
of (a) how much of the retrieved context overlaps with the cache, and (b) how
much per-chunk recompute ("blending") is spent repairing cross-chunk
attention.

Reported per configuration:
  kl          mean KL(exact || cached) over the continuation's next-token
              distributions — the fidelity cost of reuse
  top1        fraction of continuation positions where cached and exact
              prefills agree on the argmax token
  compute     fraction of prompt tokens that had to run through the model
              (fresh + blended); 1 - compute is the prefill saving
  prefill_ms  wall-clock assembly time

Usage:
  python benchmarks/divergence.py --tiny                 # random weights, smoke
  python benchmarks/divergence.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import torch

from semantic_kv import Chunk, CachedRAGPipeline, InMemoryVectorStore, KVStore, SemanticKVCache

TOPICS = [
    ("rivers", "Rivers deposit sediment where the current slows, building deltas over centuries."),
    ("glaciers", "Glacial melt drives seasonal variation in downstream flow and water temperature."),
    ("soil", "Soil cores preserve layered records of flood frequency and drought in silt bands."),
    ("forests", "Forest canopies intercept rainfall and moderate the erosion of exposed slopes."),
    ("wetlands", "Wetlands buffer storm surge and filter nitrogen runoff before it reaches the coast."),
    ("aquifers", "Aquifer recharge depends on permeable ground and multi-year precipitation trends."),
    ("estuaries", "Estuaries mix fresh and salt water, creating gradients that shape species ranges."),
    ("permafrost", "Thawing permafrost releases stored carbon and destabilizes northern infrastructure."),
]

QUERY = "Summarize the main physical processes described in the context."
CONTINUATION = ("The context describes how water, sediment, and vegetation interact "
                "to shape landscapes over time.")


def build_contexts(overlap: float, n_chunks: int = 6) -> tuple[list[Chunk], list[Chunk]]:
    """A cached 'turn 1' context and a 'turn 2' context sharing `overlap` of its chunks."""
    base = [Chunk(text, doc_id=topic) for topic, text in TOPICS[:n_chunks]]
    n_shared = round(overlap * n_chunks)
    replacements = [Chunk(text, doc_id=topic) for topic, text in TOPICS[n_chunks:]]
    turn2 = base[:n_shared] + replacements[: n_chunks - n_shared]
    return base, turn2


def run_config(pipe: CachedRAGPipeline, tokenizer, overlap: float, ratio: float) -> dict:
    turn1, turn2 = build_contexts(overlap)
    pipe.cache.kv = KVStore(max_bytes=pipe.cache.kv.max_bytes)
    pipe.cache.vectors = InMemoryVectorStore()
    pipe.cache.warm(turn1)

    cont_ids = tokenizer(CONTINUATION, add_special_tokens=False)["input_ids"]

    t0 = time.perf_counter()
    asm = pipe.assemble_for_query(QUERY, turn2, recompute_ratio=ratio)
    prefill_ms = (time.perf_counter() - t0) * 1000

    cached_logits = pipe.score_continuation(QUERY, turn2, cont_ids, recompute_ratio=ratio)
    exact_logits = pipe.score_continuation(QUERY, turn2, cont_ids, recompute_ratio=1.0)

    lp_cached = torch.log_softmax(cached_logits.float(), dim=-1)
    lp_exact = torch.log_softmax(exact_logits.float(), dim=-1)
    kl = torch.nn.functional.kl_div(lp_cached, lp_exact, log_target=True,
                                    reduction="none").sum(-1).mean().item()
    top1 = (cached_logits.argmax(-1) == exact_logits.argmax(-1)).float().mean().item()

    s = asm.stats
    compute = (s.fresh_tokens + s.blended_tokens) / s.prompt_tokens
    return {
        "overlap": overlap,
        "recompute_ratio": ratio,
        "kl": round(kl, 5),
        "top1": round(top1, 4),
        "compute": round(compute, 4),
        "saving": round(1 - compute, 4),
        "prefill_ms": round(prefill_ms, 1),
        "prompt_tokens": s.prompt_tokens,
        "hits": s.exact_hits + s.semantic_hits,
        "misses": s.misses,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tiny", action="store_true",
                    help="random-weight tiny Llama + GPT-2 tokenizer (offline smoke run)")
    ap.add_argument("--overlaps", default="1.0,0.8,0.5")
    ap.add_argument("--ratios", default="0.0,0.15,0.3,1.0")
    ap.add_argument("--out", default="results/divergence.json")
    args = ap.parse_args()

    if args.tiny:
        from transformers import GPT2TokenizerFast, LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(
            hidden_size=256, intermediate_size=512, num_hidden_layers=4,
            num_attention_heads=8, num_key_value_heads=4, vocab_size=50257,
        )).eval()
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        model_name = "tiny-random-llama (untrained — fidelity numbers are illustrative only)"
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
        model_name = args.model

    cache = SemanticKVCache(model, tokenizer, similarity_threshold=0.85)
    pipe = CachedRAGPipeline(cache)

    rows = []
    overlaps = [float(x) for x in args.overlaps.split(",")]
    ratios = [float(x) for x in args.ratios.split(",")]
    for overlap, ratio in itertools.product(overlaps, ratios):
        row = run_config(pipe, tokenizer, overlap, ratio)
        rows.append(row)
        print(f"overlap={overlap:.2f} ratio={ratio:.2f} -> kl={row['kl']:.4f} "
              f"top1={row['top1']:.3f} saving={row['saving']:.1%} "
              f"prefill={row['prefill_ms']:.0f}ms")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model_name, "rows": rows}, indent=2))
    print(f"\nwrote {out}")

    print(f"\n| overlap | blend ratio | KL | top-1 agree | prefill saving |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['overlap']:.0%} | {r['recompute_ratio']:.2f} | {r['kl']:.4f} "
              f"| {r['top1']:.1%} | {r['saving']:.1%} |")


if __name__ == "__main__":
    main()
