"""Token selection utilities for synthetic dictionary tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from transformers import PreTrainedTokenizerBase


_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{1,16}$")


@dataclass(frozen=True)
class TokenCandidate:
    """Clean candidate token extracted from GPT-2 vocabulary."""

    token_id: int
    raw_token: str
    decoded: str


def _is_valid_decoded(decoded: str) -> bool:
    stripped = decoded.strip()
    if not stripped:
        return False
    if any(ord(ch) < 32 for ch in stripped):
        return False
    return bool(_WORD_RE.fullmatch(stripped))


def _iter_candidates(
    tokenizer: PreTrainedTokenizerBase,
    require_space_prefix: bool,
) -> Iterable[TokenCandidate]:
    vocab = tokenizer.get_vocab()
    id_to_token = {idx: token for token, idx in vocab.items()}

    for token_id in range(len(id_to_token)):
        raw = id_to_token.get(token_id)
        if raw is None:
            continue
        if require_space_prefix and not raw.startswith("Ġ"):
            continue

        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if not _is_valid_decoded(decoded):
            continue
        yield TokenCandidate(token_id=token_id, raw_token=raw, decoded=decoded)


def select_clean_tokens(
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    seed: int = 0,
    require_space_prefix: bool = True,
) -> list[str]:
    """Select clean textual tokens that map to one GPT-2 token.

    Returned strings preserve leading spaces in decoded form, making them safe for
    next-token objectives after punctuation templates like ``A:``.
    """
    candidates = list(_iter_candidates(tokenizer, require_space_prefix))
    if len(candidates) < n:
        fallback = list(_iter_candidates(tokenizer, require_space_prefix=False))
        candidates = candidates + [c for c in fallback if c not in candidates]

    unique_by_text: dict[str, TokenCandidate] = {}
    for candidate in candidates:
        unique_by_text.setdefault(candidate.decoded, candidate)
    candidates = list(unique_by_text.values())

    if len(candidates) < n:
        raise ValueError(
            f"Not enough clean tokens ({len(candidates)}) to sample {n}."
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[int(i)].decoded for i in indices]


def ensure_answer_token_length(
    tokenizer: PreTrainedTokenizerBase,
    value: str,
    max_tokens: int = 2,
) -> list[int]:
    """Tokenize answer value and enforce max token count."""
    token_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
    if not token_ids or len(token_ids) > max_tokens:
        raise ValueError(
            f"Answer '{value}' tokenized to {len(token_ids)} tokens (max {max_tokens})."
        )
    return token_ids
