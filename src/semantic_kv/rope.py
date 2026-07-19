"""RoPE re-rotation: move cached keys to a new absolute position.

Rotary position embeddings apply a block-diagonal rotation R(p) to queries and
keys at position p. Rotations compose: R(p + d) = R(d) @ R(p). So keys cached
at positions [p0, p0+L) can be moved to [q0, q0+L) *exactly* by applying
R(q0 - p0) once — the positional component of KV reuse is not approximate.

What re-rotation cannot fix is the contextual component: a cached key is a
projection of a hidden state that attended only to the tokens present when it
was encoded. That residual error is the object of study in benchmarks/.

Values carry no positional encoding under RoPE and are reused as-is.
"""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rerotate_keys(keys: torch.Tensor, delta: int, inv_freq: torch.Tensor) -> torch.Tensor:
    """Shift the RoPE phase of `keys` by `delta` positions.

    keys: (batch, kv_heads, seq, head_dim), already rotated at their original
    positions. inv_freq: (head_dim/2,) from the model's rotary embedding.
    Returns keys as if they had been rotated at original_position + delta.
    """
    if delta == 0:
        return keys
    freqs = inv_freq.to(dtype=torch.float32) * float(delta)  # (head_dim/2,)
    emb = torch.cat((freqs, freqs), dim=-1)  # HF rotate_half layout
    cos = emb.cos().to(keys.dtype).to(keys.device)
    sin = emb.sin().to(keys.dtype).to(keys.device)
    return keys * cos + rotate_half(keys) * sin


def get_inv_freq(model) -> torch.Tensor:
    """Locate the rotary inv_freq buffer on a HF causal LM (Llama/Qwen/Mistral-style)."""
    for attr in ("model", "transformer"):
        base = getattr(model, attr, None)
        if base is not None and hasattr(base, "rotary_emb"):
            return base.rotary_emb.inv_freq
    if hasattr(model, "rotary_emb"):
        return model.rotary_emb.inv_freq
    raise AttributeError(
        f"Could not find rotary_emb.inv_freq on {type(model).__name__}; "
        "semantic-kv requires a RoPE-based model (Llama/Qwen/Mistral family)."
    )
