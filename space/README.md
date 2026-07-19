---
title: Semantic-diff KV Cache for RAG
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# semantic-kv demo

Interactive demo of a prefix-caching layer keyed on **semantic similarity** of
retrieved chunks. Ask related questions back-to-back: overlapping retrieved
context is served from a chunk-level KV cache (RoPE re-rotated to its new
position), and only the delta is computed. The blend slider controls how much
of each reused chunk is recomputed against its true prefix — 1.0 is exact.

Source, tests, and the divergence benchmark:
https://github.com/aabhimittal/semantic-diff-kv-cache-rag

## Deploying this Space

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
hf repo create semantic-kv-demo --repo-type space --space-sdk gradio
# copy space/* plus the package source into the space repo:
hf upload <user>/semantic-kv-demo space . --repo-type space
```

The Space needs the `semantic_kv` package; `requirements.txt` installs it
straight from GitHub.
