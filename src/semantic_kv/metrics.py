"""Per-request cache telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AssemblyStats:
    """What happened while assembling one prompt's KV."""

    prompt_tokens: int = 0
    reused_tokens: int = 0        # tokens whose KV came straight from cache (re-rotated)
    blended_tokens: int = 0       # tokens inside reused segments recomputed with true prefix
    fresh_tokens: int = 0         # prefix/suffix/miss tokens computed normally
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    substitutions: list[dict] = field(default_factory=list)  # semantic hits: what replaced what

    @property
    def reuse_fraction(self) -> float:
        return self.reused_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "reused_tokens": self.reused_tokens,
            "blended_tokens": self.blended_tokens,
            "fresh_tokens": self.fresh_tokens,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "reuse_fraction": round(self.reuse_fraction, 4),
        }
