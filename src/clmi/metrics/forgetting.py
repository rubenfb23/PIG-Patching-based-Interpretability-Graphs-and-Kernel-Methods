"""Continual learning forgetting and hysteresis metrics."""

from __future__ import annotations

import torch
import numpy as np
from transformers import PreTrainedModel


def compute_forgetting(acc_a_ma: float, acc_a_mab: float) -> float:
    """Forgetting on task A after learning B."""
    return float(acc_a_ma - acc_a_mab)


@torch.no_grad()
def compute_weight_distance(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
) -> dict[str, float]:
    """L2 and cosine distance between two model parameter states."""
    flat_a: list[torch.Tensor] = []
    flat_b: list[torch.Tensor] = []
    for (_, pa), (_, pb) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        flat_a.append(pa.detach().reshape(-1).float())
        flat_b.append(pb.detach().reshape(-1).float())
    va = torch.cat(flat_a)
    vb = torch.cat(flat_b)
    l2 = float(torch.norm(va - vb).item())
    denom = torch.norm(va) * torch.norm(vb)
    cosine = float(torch.dot(va, vb) / torch.clamp(denom, min=1e-12))
    return {"l2_distance": l2, "cosine_similarity": cosine}


def compute_hysteresis_metrics(
    retrain_acc_curve: list[float],
    acc_ref: float,
    threshold_ratio: float = 0.95,
) -> dict[str, float]:
    """Compute remanence, coercivity (steps), and loop area.

    Args:
        retrain_acc_curve: Accuracy trajectory during A-retraining (A->B->A).
        acc_ref: Reference accuracy (typically Acc_A(MA)).
        threshold_ratio: Target fraction of acc_ref for coercivity.
    """
    if not retrain_acc_curve:
        return {
            "remanence": 0.0,
            "coercivity_steps": float("nan"),
            "hysteresis_area": 0.0,
        }

    remanence = float(retrain_acc_curve[0])
    threshold = threshold_ratio * acc_ref

    coercivity_steps = float("nan")
    for i, acc in enumerate(retrain_acc_curve):
        if acc >= threshold:
            coercivity_steps = float(i)
            break

    deficits = np.maximum(0.0, acc_ref - np.asarray(retrain_acc_curve, dtype=float))
    area = float(np.trapezoid(deficits, dx=1.0))

    return {
        "remanence": remanence,
        "coercivity_steps": coercivity_steps,
        "hysteresis_area": area,
    }
