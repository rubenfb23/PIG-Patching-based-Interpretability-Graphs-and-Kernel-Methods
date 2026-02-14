"""Continual learning forgetting and hysteresis metrics."""

from __future__ import annotations

import numpy as np


def compute_forgetting(acc_a_ma: float, acc_a_mab: float) -> float:
    """Forgetting on task A after learning B."""
    return float(acc_a_ma - acc_a_mab)


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
