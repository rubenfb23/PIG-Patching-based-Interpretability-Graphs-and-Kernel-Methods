"""Kernel definitions between task representations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def k_proj(
    projectors_a: dict[int, np.ndarray],
    projectors_b: dict[int, np.ndarray],
    alpha_layers: dict[int, float] | None = None,
) -> float:
    """Projection-overlap kernel: sum_l alpha_l * Tr(P_A P_B)."""
    shared = sorted(set(projectors_a) & set(projectors_b))
    if not shared:
        return 0.0

    alpha_layers = alpha_layers or {l: 1.0 for l in shared}
    score = 0.0
    for l in shared:
        score += float(alpha_layers.get(l, 1.0) * np.trace(projectors_a[l] @ projectors_b[l]))
    return score


def k_nc(
    projectors_a: dict[int, np.ndarray],
    projectors_b: dict[int, np.ndarray],
    gamma: float,
    alpha_layers: dict[int, float] | None = None,
) -> float:
    """Commutator kernel: exp(-gamma * sum_l alpha_l ||[P_A,P_B]||_F^2)."""
    shared = sorted(set(projectors_a) & set(projectors_b))
    if not shared:
        return 0.0

    alpha_layers = alpha_layers or {l: 1.0 for l in shared}
    value = 0.0
    for l in shared:
        comm = projectors_a[l] @ projectors_b[l] - projectors_b[l] @ projectors_a[l]
        value += float(alpha_layers.get(l, 1.0) * (np.linalg.norm(comm, ord="fro") ** 2))
    return float(np.exp(-gamma * value))


def k_func(
    phi_a: np.ndarray,
    phi_b: np.ndarray,
    kind: str = "linear",
    gamma: float = 1.0,
) -> float:
    """Functional kernel on intervention embeddings phi(T)."""
    if kind == "linear":
        return float(np.dot(phi_a, phi_b))
    if kind == "rbf":
        diff = phi_a - phi_b
        return float(np.exp(-gamma * np.dot(diff, diff)))
    raise ValueError(f"Unknown kernel kind: {kind}")


def build_kernel_matrix(
    items: list[str],
    kernel_fn: Callable[[str, str], float],
) -> np.ndarray:
    """Construct symmetric kernel matrix for identifiers and pairwise kernel fn."""
    n = len(items)
    K = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            value = kernel_fn(items[i], items[j])
            K[i, j] = value
            K[j, i] = value
    return K
