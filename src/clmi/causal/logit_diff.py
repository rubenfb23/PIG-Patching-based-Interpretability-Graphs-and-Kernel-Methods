"""Logit-difference target definitions."""

from __future__ import annotations

import numpy as np


def compute_logit_diff(
    logits: np.ndarray,
    pos_token_ids: np.ndarray,
    neg_token_ids: np.ndarray,
) -> np.ndarray:
    """Compute y_i = logit(t+) - logit(t-) for each sample.

    Args:
        logits: Array with shape ``[n_samples, vocab]``.
        pos_token_ids: Correct token ids ``[n_samples]``.
        neg_token_ids: Negative token ids ``[n_samples]``.
    """
    idx = np.arange(logits.shape[0])
    return logits[idx, pos_token_ids] - logits[idx, neg_token_ids]
