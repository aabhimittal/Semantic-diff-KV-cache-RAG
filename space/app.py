"""Gradio demo for semantic-kv: watch KV reuse happen across RAG turns.

Runs a small RoPE model on CPU. Each query retrieves top-k chunks from a toy
corpus; the cache panel shows which chunks were served from KV cache (exact or
semantic), which were blended, and the prefill tokens saved.
"""

import numpy as np
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

from semantic_kv import Chunk, CachedRAGPipeline, HashingEmbedder, SemanticKVCache

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

CORPUS = {
    "refunds": "Refunds are issued to the original payment method within 5 business days "
               "of the return being received and inspected at the warehouse.",
    "shipping": "Standard shipping takes 3 to 7 business days; expedited orders placed "
                "before noon ship the same day from the nearest fulfillment center.",
    "warranty": "All devices carry a two-year limited warranty covering manufacturing "
                "defects but not accidental damage or unauthorized repairs.",
    "returns": "Items may be returned within 30 days of delivery in original packaging; "
               "opened software and gift cards are non-returnable.",
    "support": "Technical support is available by chat around the clock and by phone "
               "on weekdays; enterprise customers receive a dedicated account engineer.",
    "privacy": "Customer data is retained for two years after account closure and is "
               "never sold to third parties under any circumstances.",
}

print("loading model…")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).eval()

embedder = HashingEmbedder()
cache = SemanticKVCache(model, tok, embedder=embedder,
                        similarity_threshold=0.80, recompute_ratio=0.15)
pipe = CachedRAGPipeline(cache)

_chunks = [Chunk(text, doc_id=doc) for doc, text in CORPUS.items()]
_vecs = np.stack([embedder.embed(c.text) for c in _chunks])


def retrieve(query: str, k: int = 3) -> list[Chunk]:
    scores = _vecs @ embedder.embed(query)
    return [_chunks[i] for i in np.argsort(-scores)[:k]]


def answer(query: str, blend: float):
    if not query.strip():
        return "", "ask something!"
    chunks = retrieve(query)
    res = pipe.generate(query, chunks, max_new_tokens=80, recompute_ratio=blend)
    s = res.stats
    lines = [
        f"**prompt tokens:** {s.prompt_tokens}  |  "
        f"**reused from KV cache:** {s.reused_tokens}  |  "
        f"**blended (repaired):** {s.blended_tokens}  |  "
        f"**computed fresh:** {s.fresh_tokens}",
        f"**prefill saved:** {s.reuse_fraction:.0%}  |  "
        f"exact hits: {s.exact_hits} · semantic hits: {s.semantic_hits} · misses: {s.misses}",
    ]
    for sub in s.substitutions:
        lines.append(f"- semantic substitution (cos {sub['score']}): served “{sub['served']}…”")
    lines.append("\n**retrieved:** " + ", ".join(c.doc_id for c in chunks))
    return res.text, "\n\n".join(lines)


def reset():
    global cache, pipe
    cache = SemanticKVCache(model, tok, embedder=embedder,
                            similarity_threshold=0.80, recompute_ratio=0.15)
    pipe = CachedRAGPipeline(cache)
    return "cache cleared — next query pays full prefill"


with gr.Blocks(title="semantic-kv demo") as demo:
    gr.Markdown(
        "# semantic-kv: semantic-diff KV cache for RAG\n"
        "Ask related questions back-to-back and watch the second query reuse "
        "cached KV for overlapping retrieved chunks. "
        "[Code & benchmarks](https://github.com/aabhimittal/semantic-diff-kv-cache-rag)"
    )
    with gr.Row():
        query = gr.Textbox(label="question", placeholder="How long do refunds take?")
        blend = gr.Slider(0.0, 1.0, value=0.15, step=0.05,
                          label="blend ratio (fraction of each reused chunk recomputed; 1.0 = exact)")
    go = gr.Button("ask", variant="primary")
    out = gr.Textbox(label="answer")
    stats = gr.Markdown(label="cache telemetry")
    clear = gr.Button("reset cache")
    go.click(answer, [query, blend], [out, stats])
    query.submit(answer, [query, blend], [out, stats])
    clear.click(reset, None, stats)

if __name__ == "__main__":
    demo.launch()
