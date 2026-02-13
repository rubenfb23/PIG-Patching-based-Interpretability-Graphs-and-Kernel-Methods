"""Graph strategy using absolute correlation values before sparsification."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pig.graph import GraphBuilder
from pig.graphs.registry import register_graph_builder


class AbsoluteCorrelationGraphBuilder(GraphBuilder):
    """Top-k graph builder that uses absolute pairwise correlations."""

    def _compute_correlation_matrix(
        self,
        effect_matrix: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        corr = super()._compute_correlation_matrix(effect_matrix)
        return np.abs(corr).astype(np.float32)


@register_graph_builder("abs_correlation_topk")
def create_builder(**kwargs):
    """Create an absolute-correlation top-k graph builder."""
    return AbsoluteCorrelationGraphBuilder(**kwargs)
