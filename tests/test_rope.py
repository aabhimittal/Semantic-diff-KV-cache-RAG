import torch

from semantic_kv.rope import get_inv_freq, rerotate_keys, rotate_half


def _apply_rope(x, positions, inv_freq):
    freqs = torch.outer(positions.float(), inv_freq)  # (seq, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return x * emb.cos() + rotate_half(x) * emb.sin()


def test_rerotation_matches_direct_rope(model):
    """R(delta) applied to keys rotated at p must equal keys rotated at p+delta."""
    inv_freq = get_inv_freq(model)
    torch.manual_seed(1)
    raw = torch.randn(1, 2, 6, inv_freq.numel() * 2)
    p = torch.arange(1, 7)
    delta = 37
    at_p = _apply_rope(raw, p, inv_freq)
    shifted = rerotate_keys(at_p, delta, inv_freq)
    direct = _apply_rope(raw, p + delta, inv_freq)
    assert torch.allclose(shifted, direct, atol=1e-5)


def test_zero_delta_is_identity(model):
    inv_freq = get_inv_freq(model)
    k = torch.randn(1, 2, 4, inv_freq.numel() * 2)
    assert rerotate_keys(k, 0, inv_freq) is k
