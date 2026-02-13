"""Low-level architecture and tokenization helpers for the toy model."""

from __future__ import annotations

import hashlib
import re

from torch import nn

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


class TinyLayer(nn.Module):
    """Single pre-norm transformer block."""

    def __init__(self, d_model: int, n_heads: int, mlp_dim: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ln_2 = nn.LayerNorm(d_model)
        self.fc_1 = nn.Linear(d_model, mlp_dim)
        self.fc_2 = nn.Linear(mlp_dim, d_model)

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads


def split_tokens(text: str) -> list[str]:
    """Split text into word/punctuation tokens."""
    tokens = TOKEN_PATTERN.findall(text)
    return tokens if tokens else ["<empty>"]


def token_to_hashed_id(token: str, vocab_size: int) -> int:
    """Map a token to a stable hashed vocabulary ID."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return 2 + value % (vocab_size - 2)
