"""Shared torch utilities for CLMI modules."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel


def unwrap_model(model: PreTrainedModel) -> PreTrainedModel:
    """Return the underlying base model, unwrapping PeftModel layers if present."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def model_dtype(model: PreTrainedModel) -> torch.dtype:
    """Infer the floating-point dtype from the first model parameter."""
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute KL(p || q) row-wise.  Returns shape ``[batch]``."""
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    return torch.sum(p_safe * (torch.log(p_safe) - torch.log(q_safe)), dim=-1)
