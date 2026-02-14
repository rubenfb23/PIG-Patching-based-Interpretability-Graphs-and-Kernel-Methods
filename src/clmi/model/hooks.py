"""Activation capture and projection intervention helpers."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from clmi.utils.torch_helpers import model_dtype, unwrap_model


InterventionOrder = Literal["AB", "BA"]
InterventionMode = Literal["reinforce", "suppress"]


def _get_blocks(model: PreTrainedModel):
    return unwrap_model(model).transformer.h


@torch.no_grad()
def capture_residual_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    layers: list[int],
    device: str,
) -> dict[int, torch.Tensor]:
    """Capture per-layer residual activations at the final prompt token."""
    model.eval()
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    outputs = model(**tokens, output_hidden_states=True, use_cache=False)
    attn = tokens["attention_mask"]
    last_pos = attn.sum(dim=1) - 1

    captured: dict[int, torch.Tensor] = {}
    for layer in layers:
        hidden = outputs.hidden_states[layer + 1]
        idx = torch.arange(hidden.shape[0], device=device)
        captured[layer] = hidden[idx, last_pos, :].detach().clone()
    return captured


def _apply_projection(
    hidden: torch.Tensor,
    projector: torch.Tensor,
    beta: float,
    mode: InterventionMode,
) -> torch.Tensor:
    # Projectors are symmetric (P = U @ U.T), so P.T == P; use P directly.
    projected = hidden @ projector
    if mode == "reinforce":
        return hidden + beta * projected
    return hidden - beta * projected


@torch.no_grad()
def run_with_projection_intervention(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    projectors_a: dict[int, torch.Tensor],
    projectors_b: dict[int, torch.Tensor] | None,
    beta: float,
    mode: InterventionMode,
    order: InterventionOrder,
    device: str,
) -> torch.Tensor:
    """Run model with soft projection interventions and return next-token probs.

    Args:
        projectors_a: Layer -> projection matrix for task A.
        projectors_b: Layer -> projection matrix for task B or None.
        order: "AB" applies A then B, "BA" applies B then A.
    """
    model.eval()
    base = unwrap_model(model)
    blocks = _get_blocks(model)
    dtype = model_dtype(model)

    tensor_a = {
        l: p.to(device=device, dtype=dtype)
        for l, p in projectors_a.items()
    }
    tensor_b = (
        {
            l: p.to(
                device=device,
                dtype=dtype,
            )
            for l, p in projectors_b.items()
        }
        if projectors_b
        else {}
    )

    hooks = []

    def _mk_hook(layer: int):
        p_a = tensor_a.get(layer)
        p_b = tensor_b.get(layer)

        def _hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                tail = output[1:]
            else:
                hidden = output
                tail = None

            if order == "AB":
                if p_a is not None:
                    hidden = _apply_projection(hidden, p_a, beta, mode)
                if p_b is not None:
                    hidden = _apply_projection(hidden, p_b, beta, mode)
            else:
                if p_b is not None:
                    hidden = _apply_projection(hidden, p_b, beta, mode)
                if p_a is not None:
                    hidden = _apply_projection(hidden, p_a, beta, mode)

            if tail is None:
                return hidden
            return (hidden,) + tail

        return _hook

    touched_layers = sorted(set(tensor_a.keys()) | set(tensor_b.keys()))
    with ExitStack() as stack:
        for layer in touched_layers:
            hooks.append(blocks[layer].register_forward_hook(_mk_hook(layer)))
            stack.callback(hooks[-1].remove)

        tokens = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        logits = model(**tokens, use_cache=False).logits
        last_idx = tokens["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(logits.shape[0], device=device)
        next_logits = logits[batch_idx, last_idx, :]
        probs = torch.softmax(next_logits, dim=-1)

    return probs
