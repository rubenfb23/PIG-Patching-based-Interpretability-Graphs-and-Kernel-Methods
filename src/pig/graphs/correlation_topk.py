"""Default graph strategy based on correlation + top-k sparsification."""

from __future__ import annotations

from pig.graph import GraphBuilder
from pig.graphs.registry import register_graph_builder


@register_graph_builder("correlation_topk")
def create_builder(**kwargs):
    """Create the default correlation-topk graph builder."""
    return GraphBuilder(**kwargs)
