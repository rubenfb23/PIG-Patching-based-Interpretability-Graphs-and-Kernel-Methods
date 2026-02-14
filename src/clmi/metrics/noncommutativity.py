"""Operator and functional non-commutativity metrics."""

from __future__ import annotations

import numpy as np
import torch

from clmi.utils.torch_helpers import kl_divergence


def compute_operator_noncommutativity(
    projectors_a: dict[int, np.ndarray],
    projectors_b: dict[int, np.ndarray],
    alpha_layers: dict[int, float] | None = None,
) -> dict[str, float | dict[int, float]]:
    """Compute layer-wise and global ||[P_A, P_B]||_F metrics."""
    shared_layers = sorted(set(projectors_a.keys()) & set(projectors_b.keys()))
    if not shared_layers:
        return {"nc_global": 0.0, "mean_nc_layers": 0.0, "nc_layers": {}}

    alpha_layers = alpha_layers or {layer: 1.0 for layer in shared_layers}

    per_layer: dict[int, float] = {}
    weighted_total = 0.0
    weight_sum = 0.0

    for layer in shared_layers:
        p_a = projectors_a[layer]
        p_b = projectors_b[layer]
        comm = p_a @ p_b - p_b @ p_a
        nc = float(np.linalg.norm(comm, ord="fro"))
        per_layer[layer] = nc

        w = float(alpha_layers.get(layer, 1.0))
        weighted_total += w * nc
        weight_sum += w

    mean_nc = float(np.mean(list(per_layer.values())))
    nc_global = float(weighted_total / max(weight_sum, 1e-8))
    return {
        "nc_global": nc_global,
        "mean_nc_layers": mean_nc,
        "nc_layers": per_layer,
    }


def compute_prob_kl(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute mean KL(p || q) over batch."""
    return float(kl_divergence(p, q, eps=eps).mean().item())


def compute_jensen_shannon(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute mean Jensen-Shannon divergence over batch.

    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m),  m = 0.5*(p+q).
    JSD is a proper symmetric metric (square root is a distance).
    """
    m = 0.5 * (p + q)
    kl_pm = kl_divergence(p, m, eps=eps)
    kl_qm = kl_divergence(q, m, eps=eps)
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return float(jsd.mean().item())
