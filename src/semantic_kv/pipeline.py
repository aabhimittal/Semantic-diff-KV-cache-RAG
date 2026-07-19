"""RAG generation on top of SemanticKVCache."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache import Assembled, SemanticKVCache
from .chunking import Chunk
from .metrics import AssemblyStats


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    stats: AssemblyStats


class CachedRAGPipeline:
    """Assemble (prefix + retrieved chunks + query) with KV reuse, then decode."""

    def __init__(self, cache: SemanticKVCache,
                 prefix_template: str = "Use the following context to answer.\n\n",
                 query_template: str = "\n\nQuestion: {query}\nAnswer:"):
        self.cache = cache
        self.prefix_template = prefix_template
        self.query_template = query_template

    def assemble_for_query(self, query: str, chunks: list[Chunk],
                           recompute_ratio: float | None = None) -> Assembled:
        return self.cache.assemble(
            chunks,
            prefix_text=self.prefix_template,
            suffix_text=self.query_template.format(query=query),
            recompute_ratio=recompute_ratio,
        )

    @torch.no_grad()
    def generate(self, query: str, chunks: list[Chunk], max_new_tokens: int = 64,
                 temperature: float = 0.0,
                 recompute_ratio: float | None = None) -> GenerationResult:
        asm = self.assemble_for_query(query, chunks, recompute_ratio=recompute_ratio)
        model, tok = self.cache.model, self.cache.tokenizer
        device = self.cache.device
        eos = tok.eos_token_id
        pos = len(asm.input_ids)
        logits = asm.last_logits
        out_ids: list[int] = []
        for _ in range(max_new_tokens):
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, 1))
            else:
                next_id = int(torch.argmax(logits))
            if eos is not None and next_id == eos:
                break
            out_ids.append(next_id)
            step = model(
                torch.tensor([[next_id]], device=device),
                past_key_values=asm.cache,
                use_cache=True,
                attention_mask=torch.ones(1, pos + 1, dtype=torch.long, device=device),
                position_ids=torch.tensor([[pos]], device=device),
                cache_position=torch.tensor([pos], device=device),
            )
            logits = step.logits[0, -1]
            pos += 1
        return GenerationResult(
            text=tok.decode(out_ids, skip_special_tokens=True),
            token_ids=out_ids,
            stats=asm.stats,
        )

    @torch.no_grad()
    def score_continuation(self, query: str, chunks: list[Chunk], continuation_ids: list[int],
                           recompute_ratio: float | None = None) -> torch.Tensor:
        """Logits over each position of `continuation_ids` given the assembled prompt.

        Returns (len(continuation_ids), vocab): position i holds the logits
        *predicting* continuation_ids[i]. Used by the divergence benchmark to
        compare cached-vs-exact prefills on identical continuations.
        """
        asm = self.assemble_for_query(query, chunks, recompute_ratio=recompute_ratio)
        rows = [asm.last_logits]
        if len(continuation_ids) > 1:
            pos = len(asm.input_ids)
            n = len(continuation_ids) - 1
            device = self.cache.device
            out = self.cache.model(
                torch.tensor([continuation_ids[:-1]], device=device),
                past_key_values=asm.cache,
                use_cache=True,
                attention_mask=torch.ones(1, pos + n, dtype=torch.long, device=device),
                position_ids=torch.arange(pos, pos + n, device=device)[None],
                cache_position=torch.arange(pos, pos + n, device=device),
            )
            rows.extend(out.logits[0])
        return torch.stack(rows)
