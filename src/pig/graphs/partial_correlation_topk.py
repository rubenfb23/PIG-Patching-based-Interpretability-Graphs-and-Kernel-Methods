"""Graph strategy using partial correlations from a regularized precision matrix."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pig.graph import GraphBuilder
from pig.graphs.registry import register_graph_builder


class PartialCorrelationGraphBuilder(GraphBuilder):
    """Top-k graph builder based on conditional co-variation.

    The default correlation graph connects nodes whose patch-effect profiles
    co-vary marginally. This builder instead estimates a precision matrix over
    node effects and converts it to signed partial correlations, so an edge
    weight measures residual association after conditioning on the other nodes.
    A small ridge term keeps the estimate defined in the common high-dimensional
    bootstrap setting where the number of nodes exceeds the number of examples.
    """

    def __init__(
        self,
        k: int = 5,
        enforce_direction: bool = True,
        min_weight: float = 0.0,
        ridge: float = 1e-3,
    ):
        super().__init__(
            k=k,
            enforce_direction=enforce_direction,
            min_weight=min_weight,
        )
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        self.ridge = ridge

    def _compute_correlation_matrix(
        self,
        effect_matrix: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        centered = effect_matrix - effect_matrix.mean(axis=0, keepdims=True)
        n_samples = max(int(centered.shape[0]), 1)
        covariance = (centered.T @ centered) / n_samples

        scale = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
        covariance = covariance / np.outer(scale, scale)
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)

        if self.ridge > 0:
            covariance = covariance + np.eye(covariance.shape[0]) * self.ridge

        precision = np.linalg.pinv(covariance, hermitian=True)
        diagonal = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
        partial = -precision / np.outer(diagonal, diagonal)
        np.fill_diagonal(partial, 0.0)
        return np.nan_to_num(partial, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

    def build_from_tensors(self, *args, **kwargs):
        kwargs.setdefault("construction_mode", "slice_partial_correlation")
        graph = super().build_from_tensors(*args, **kwargs)
        graph.metadata["ridge"] = float(self.ridge)
        return graph


@register_graph_builder("partial_correlation_topk")
def create_builder(**kwargs):
    """Create a partial-correlation top-k graph builder."""
    return PartialCorrelationGraphBuilder(**kwargs)
