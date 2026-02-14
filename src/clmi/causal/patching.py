"""Functional intervention utilities (AB vs BA) for non-commutativity analysis."""

from __future__ import annotations

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from clmi.model.hooks import run_with_projection_intervention
from clmi.utils.torch_helpers import kl_divergence


@torch.no_grad()
def compute_functional_kl_ab_ba(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    projectors_a: dict[int, np.ndarray],
    projectors_b: dict[int, np.ndarray],
    beta: float,
    mode: str,
    device: str,
) -> float:
    """Compute KL(p_AB || p_BA) averaged over prompts."""
    torch_a = {layer: torch.from_numpy(p) for layer, p in projectors_a.items()}
    torch_b = {layer: torch.from_numpy(p) for layer, p in projectors_b.items()}

    probs_ab = run_with_projection_intervention(
        model,
        tokenizer,
        prompts,
        projectors_a=torch_a,
        projectors_b=torch_b,
        beta=beta,
        mode=mode,
        order="AB",
        device=device,
    )
    probs_ba = run_with_projection_intervention(
        model,
        tokenizer,
        prompts,
        projectors_a=torch_a,
        projectors_b=torch_b,
        beta=beta,
        mode=mode,
        order="BA",
        device=device,
    )

    return float(kl_divergence(probs_ab, probs_ba).mean().item())


@torch.no_grad()
def compute_intervention_effect_embedding(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    positive_token_ids: list[int],
    negative_token_ids: list[int],
    projectors: dict[int, np.ndarray],
    beta: float,
    mode: str,
    device: str,
) -> np.ndarray:
    """Build phi(T): intervention effect vector on a probe prompt set."""
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    base_logits = model(**tokens, use_cache=False).logits
    last_idx = tokens["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(base_logits.shape[0], device=device)
    base_next = base_logits[batch_idx, last_idx, :]

    # Compute baseline log-probs from logits for a consistent comparison.
    base_logprobs = torch.log_softmax(base_next, dim=-1)

    torch_proj = {layer: torch.from_numpy(p) for layer, p in projectors.items()}
    probs_intervened = run_with_projection_intervention(
        model,
        tokenizer,
        prompts,
        projectors_a=torch_proj,
        projectors_b=None,
        beta=beta,
        mode=mode,
        order="AB",
        device=device,
    )

    inter_logprobs = torch.log(torch.clamp(probs_intervened, min=1e-8))

    pos = torch.tensor(positive_token_ids, device=device)
    neg = torch.tensor(negative_token_ids, device=device)

    base_ld = base_logprobs[batch_idx, pos] - base_logprobs[batch_idx, neg]
    inter_ld = inter_logprobs[batch_idx, pos] - inter_logprobs[batch_idx, neg]
    delta = inter_ld - base_ld

    return delta.detach().cpu().numpy().astype(np.float32)
