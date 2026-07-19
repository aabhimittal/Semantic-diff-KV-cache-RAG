import pytest
import torch
from transformers import GPT2TokenizerFast, LlamaConfig, LlamaForCausalLM


@pytest.fixture(scope="session")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=50257,
        max_position_embeddings=2048,
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="session")
def tokenizer():
    # A tiny local BPE tokenizer built from scratch would be ideal; GPT-2's is
    # small and cached by HF. Fall back to a whitespace codec offline.
    try:
        return GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception:
        return _WhitespaceTokenizer()


class _WhitespaceTokenizer:
    """Offline fallback: hash words into the vocab range deterministically."""

    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        import hashlib

        def wid(w):
            return 3 + int.from_bytes(hashlib.md5(w.encode()).digest()[:4], "little") % 50000

        return {"input_ids": [wid(w) for w in text.split()]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"<{i}>" for i in ids)
