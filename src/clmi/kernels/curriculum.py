"""Greedy curriculum ordering from pairwise kernel similarities."""

from __future__ import annotations

import numpy as np


def greedy_curriculum(
    task_ids: list[str],
    kernel_matrix: np.ndarray,
    objective: str = "min_conflict",
) -> list[str]:
    """Return greedy task order maximizing compatibility with visited tasks."""
    if kernel_matrix.shape[0] != len(task_ids):
        raise ValueError("kernel_matrix size does not match task_ids")

    n = len(task_ids)
    if n == 0:
        return []

    remaining = set(range(n))
    start = int(np.argmax(kernel_matrix.mean(axis=1)))
    order = [start]
    remaining.remove(start)

    while remaining:
        best_idx = None
        best_score = -float("inf")
        for idx in remaining:
            score = float(np.mean([kernel_matrix[idx, j] for j in order]))
            if objective == "min_conflict":
                candidate = score
            elif objective == "max_similarity":
                candidate = score
            else:
                raise ValueError(f"Unknown objective: {objective}")

            if candidate > best_score:
                best_score = candidate
                best_idx = idx

        assert best_idx is not None
        order.append(best_idx)
        remaining.remove(best_idx)

    return [task_ids[i] for i in order]
