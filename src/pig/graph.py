"""Graph construction from patch-effect tensors.

This module builds sparse directed weighted graphs from patch-effect data.
The canonical publication-facing object is the per-slice graph built by
``build_from_slice()``, where edges reflect cross-example co-variation of
patch effects. ``build_per_example()`` remains available as an auxiliary
classification baseline.

Key features:
- Fixed node set V across all slices (same patch points)
- Edge weights from correlation of effect profiles
- Direction constraint: edges from earlier (ℓ, t, component) to later
- Top-k sparsification per node
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pig.patching import ComponentSpec, PatchEffectDataset
from pig.prompts import SliceLabel

DEFAULT_GRAPH_BUILDER = "correlation_topk"
GRAPH_BUILDER_ENV_VAR = "PIG_GRAPH_BUILDER"


@dataclass(frozen=True)
class Node:
    """A node in the patch-influence graph.

    Represents a (layer, token, component) patch point.
    """

    layer: int
    token: int
    node_type: str = "res"  # "res", "mlp", or "att"
    head: Optional[int] = None

    def __lt__(self, other: "Node") -> bool:
        """Ordering for direction constraint: earlier layers/tokens first."""
        if self.layer != other.layer:
            return self.layer < other.layer
        if self.token != other.token:
            return self.token < other.token
        type_order = {"att": 0, "mlp": 1, "res": 2}
        if self.node_type != other.node_type:
            return type_order.get(self.node_type, 99) < type_order.get(
                other.node_type, 99
            )
        self_head = -1 if self.head is None else self.head
        other_head = -1 if other.head is None else other.head
        return self_head < other_head

    def to_index(
        self,
        num_tokens: int,
        component_axis: Optional[list[ComponentSpec]] = None,
    ) -> int:
        """Convert to linear index."""
        if component_axis is None:
            if self.node_type != "res" or self.head is not None:
                raise ValueError(
                    "component_axis required for non-residual nodes"
                )
            return self.layer * num_tokens + self.token
        component_idx = self._component_index(component_axis)
        return (
            (self.layer * num_tokens + self.token) * len(component_axis)
            + component_idx
        )

    def _component_index(self, component_axis: list[ComponentSpec]) -> int:
        for idx, comp in enumerate(component_axis):
            if comp.node_type == self.node_type and comp.head == self.head:
                return idx
        raise ValueError(
            f"Node not found in component_axis: {self.node_type}, {self.head}"
        )

    @classmethod
    def from_index(
        cls,
        idx: int,
        num_tokens: int,
        component_axis: Optional[list[ComponentSpec]] = None,
    ) -> "Node":
        """Create from linear index."""
        if component_axis is None:
            layer = idx // num_tokens
            token = idx % num_tokens
            return cls(layer=layer, token=token)
        components = len(component_axis)
        layer_token = idx // components
        component_idx = idx % components
        layer = layer_token // num_tokens
        token = layer_token % num_tokens
        comp = component_axis[component_idx]
        return cls(
            layer=layer,
            token=token,
            node_type=comp.node_type,
            head=comp.head,
        )


@dataclass
class Edge:
    """A directed edge in the patch-influence graph."""

    src: int  # Source node index
    dst: int  # Destination node index
    weight: float  # Edge weight (correlation)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"src": self.src, "dst": self.dst, "weight": self.weight}


@dataclass
class PatchInfluenceGraph:
    """Sparse directed weighted graph of patch influences.

    Represents the circuit structure for a single slice, where edges
    indicate co-influence relationships between patch positions.
    """

    nodes: list[Node]
    edges: list[Edge]
    slice_label: SliceLabel
    num_layers: int
    num_tokens: int
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_adjacency_matrix(self) -> NDArray[np.float32]:
        """Return dense adjacency matrix."""
        n = len(self.nodes)
        adj = np.zeros((n, n), dtype=np.float32)
        for edge in self.edges:
            adj[edge.src, edge.dst] = edge.weight
        return adj

    def get_edge_list(self) -> list[tuple[int, int, float]]:
        """Return edges as list of (src, dst, weight) tuples."""
        return [(e.src, e.dst, e.weight) for e in self.edges]

    def get_node_degrees(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Return (in_degree, out_degree) arrays."""
        n = len(self.nodes)
        in_deg = np.zeros(n, dtype=np.int32)
        out_deg = np.zeros(n, dtype=np.int32)
        for edge in self.edges:
            out_deg[edge.src] += 1
            in_deg[edge.dst] += 1
        return in_deg, out_deg

    def compute_statistics(self) -> dict:
        """Compute graph statistics."""
        in_deg, out_deg = self.get_node_degrees()
        weights = [e.weight for e in self.edges]

        stats = {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": self.num_edges / (self.num_nodes ** 2) if self.num_nodes > 0 else 0,
            "mean_in_degree": float(np.mean(in_deg)),
            "mean_out_degree": float(np.mean(out_deg)),
            "max_in_degree": int(np.max(in_deg)) if len(in_deg) > 0 else 0,
            "max_out_degree": int(np.max(out_deg)) if len(out_deg) > 0 else 0,
            "mean_weight": float(np.mean(weights)) if weights else 0,
            "std_weight": float(np.std(weights)) if weights else 0,
        }
        return stats

    def to_dict(self) -> dict:
        """Serialize to dictionary matching the schema."""
        return {
            "nodes": [
                {
                    "layer": n.layer,
                    "token": n.token,
                    "type": n.node_type,
                    **({"head": n.head} if n.head is not None else {}),
                }
                for n in self.nodes
            ],
            "edges": [e.to_dict() for e in self.edges],
            "slice": {
                "task": self.slice_label.task,
                "corruption": self.slice_label.corruption,
            },
            "metadata": self.metadata,
        }


class GraphBuilder:
    """Builds patch-influence graphs from effect tensors.

    Constructs sparse directed graphs where:
    - Nodes are (layer, token, component) positions
    - Edge weights are correlations of effect profiles
    - Direction constraint ensures information flows forward
    - Top-k sparsification limits graph density
    """

    def __init__(
        self,
        k: int = 5,
        enforce_direction: bool = True,
        min_weight: float = 0.0,
    ):
        """Initialize the graph builder.

        Args:
            k: Number of top edges to keep per source node
            enforce_direction: If True, only allow edges from earlier to later nodes
            min_weight: Minimum absolute weight to include an edge
        """
        self.k = k
        self.enforce_direction = enforce_direction
        self.min_weight = min_weight

    def _compute_correlation_matrix(
        self,
        effect_matrix: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute pairwise correlations between node effect profiles.

        Args:
            effect_matrix: Shape [num_examples, num_nodes]

        Returns:
            Correlation matrix of shape [num_nodes, num_nodes]
        """
        # Center the data
        centered = effect_matrix - effect_matrix.mean(axis=0, keepdims=True)

        # Compute standard deviations
        std = np.std(effect_matrix, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero

        # Normalize
        normalized = centered / std

        # Correlation matrix
        n_samples = effect_matrix.shape[0]
        corr = np.dot(normalized.T, normalized) / n_samples

        return corr.astype(np.float32)

    def _create_nodes(
        self,
        num_layers: int,
        num_tokens: int,
        component_axis: Optional[list[ComponentSpec]] = None,
    ) -> list[Node]:
        """Create the fixed node set."""
        nodes = []
        for layer in range(num_layers):
            for token in range(num_tokens):
                if component_axis:
                    for comp in component_axis:
                        nodes.append(
                            Node(
                                layer=layer,
                                token=token,
                                node_type=comp.node_type,
                                head=comp.head,
                            )
                        )
                else:
                    nodes.append(Node(layer=layer, token=token))
        return nodes

    def _apply_direction_constraint(
        self,
        weights: NDArray[np.float32],
        nodes: list[Node],
    ) -> NDArray[np.float32]:
        """Zero out edges that violate direction constraint."""
        n = len(nodes)
        mask = np.ones((n, n), dtype=np.float32)

        for i, src in enumerate(nodes):
            for j, dst in enumerate(nodes):
                # Only allow edges from earlier to later nodes
                if not (src < dst):
                    mask[i, j] = 0.0

        return weights * mask

    def _apply_topk_sparsification(
        self,
        weights: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Keep only top-k outgoing edges per source node."""
        n = weights.shape[0]
        sparse_weights = np.zeros_like(weights)

        for i in range(n):
            row = weights[i, :]
            # Get indices of top-k values (by absolute weight)
            if self.k >= n:
                top_indices = np.arange(n)
            else:
                top_indices = np.argsort(np.abs(row))[-self.k:]

            sparse_weights[i, top_indices] = row[top_indices]

        return sparse_weights

    def build_from_slice(
        self,
        dataset: PatchEffectDataset,
        slice_label: SliceLabel,
    ) -> PatchInfluenceGraph:
        """Build the canonical graph for a single slice.

        Args:
            dataset: Dataset containing patch-effect tensors
            slice_label: The slice to build the graph for

        Returns:
            PatchInfluenceGraph summarizing slice-level correlations
        """
        tensors = dataset.get_by_slice(slice_label)
        if not tensors:
            raise ValueError(f"No tensors found for slice {slice_label}")

        # Get common dimensions (uses minimum token count across examples)
        num_layers, num_tokens, _ = dataset.get_common_dimensions(slice_label)
        component_axis = dataset.get_component_axis()

        # Create effect matrix [num_examples, num_nodes]
        # This truncates to num_tokens to ensure consistent dimensions
        effect_matrix = dataset.get_effect_matrix(slice_label, max_tokens=num_tokens)

        # Compute correlation matrix
        corr_matrix = self._compute_correlation_matrix(effect_matrix)

        # Create nodes
        nodes = self._create_nodes(num_layers, num_tokens, component_axis)

        # Apply direction constraint
        if self.enforce_direction:
            corr_matrix = self._apply_direction_constraint(corr_matrix, nodes)

        # Apply top-k sparsification
        sparse_matrix = self._apply_topk_sparsification(corr_matrix)

        # Create edges from sparse matrix
        edges = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                weight = sparse_matrix[i, j]
                if abs(weight) > self.min_weight and i != j:
                    edges.append(Edge(src=i, dst=j, weight=float(weight)))

        return PatchInfluenceGraph(
            nodes=nodes,
            edges=edges,
            slice_label=slice_label,
            num_layers=num_layers,
            num_tokens=num_tokens,
            metadata={
                "k": self.k,
                "enforce_direction": self.enforce_direction,
                "num_examples": len(tensors),
                "graph_builder": self.__class__.__name__,
                "graph_role": "canonical_slice_graph",
                "construction_mode": "slice_correlation",
            },
        )

    def build_all(
        self,
        dataset: PatchEffectDataset,
    ) -> dict[SliceLabel, PatchInfluenceGraph]:
        """Build graphs for all slices in the dataset.

        Args:
            dataset: Dataset containing patch-effect tensors

        Returns:
            Dictionary mapping slice labels to graphs
        """
        graphs = {}
        for slice_label in dataset.get_slices():
            graphs[slice_label] = self.build_from_slice(dataset, slice_label)
        return graphs

    def build_per_example(
        self,
        dataset: PatchEffectDataset,
    ) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
        """Build one auxiliary graph per example for classification.

        These graphs are useful as an auxiliary baseline because they provide
        more samples for classifiers, but they are not the canonical method
        object used by the main PIG narrative.

        Args:
            dataset: Dataset containing patch-effect tensors

        Returns:
            List of (graph, slice_label) tuples
        """
        graphs = []

        for tensor in dataset:
            num_layers = tensor.num_layers
            num_tokens = tensor.num_tokens
            component_axis = tensor.component_axis

            # Create nodes
            nodes = self._create_nodes(
                num_layers, num_tokens, component_axis
            )

            # Build edge weights from effect magnitudes
            # Edges connect positions with similar effect magnitudes
            effects_flat = tensor.effects.reshape(-1)
            n_nodes = len(nodes)

            # Compute similarity based on effect values
            weights = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        # Similarity: high when effects are similar
                        diff = abs(effects_flat[i] - effects_flat[j])
                        weights[i, j] = 1.0 / (1.0 + diff)

            # Apply direction constraint
            if self.enforce_direction:
                weights = self._apply_direction_constraint(weights, nodes)

            # Apply top-k sparsification
            sparse_weights = self._apply_topk_sparsification(weights)

            # Create edges
            edges = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    weight = sparse_weights[i, j]
                    if abs(weight) > self.min_weight and i != j:
                        edges.append(Edge(src=i, dst=j, weight=float(weight)))

            graph = PatchInfluenceGraph(
                nodes=nodes,
                edges=edges,
                slice_label=tensor.prompt_pair.slice_label,
                num_layers=num_layers,
                num_tokens=num_tokens,
                metadata={
                    "k": self.k,
                    "example_based": True,
                    "graph_builder": self.__class__.__name__,
                    "graph_role": "auxiliary_per_example_baseline",
                    "construction_mode": "per_example_similarity",
                },
            )
            graphs.append((graph, tensor.prompt_pair.slice_label))

        return graphs


def get_available_graph_builders() -> list[str]:
    """List available graph strategy names discovered in `pig.graphs`."""
    from pig.graphs.registry import list_graph_builders

    return list_graph_builders()


def create_graph_builder(
    builder_name: str | None = None,
    *,
    k: int = 5,
    enforce_direction: bool = True,
    min_weight: float = 0.0,
) -> GraphBuilder:
    """Create a graph builder by registered strategy name.

    If `builder_name` is omitted, reads `PIG_GRAPH_BUILDER` and falls back
    to `"correlation_topk"`.
    """
    resolved_name = (
        builder_name
        or os.getenv(GRAPH_BUILDER_ENV_VAR, DEFAULT_GRAPH_BUILDER)
    ).strip().lower()
    if not resolved_name:
        resolved_name = DEFAULT_GRAPH_BUILDER

    from pig.graphs.registry import create_graph_builder as _create_graph_builder

    return _create_graph_builder(
        resolved_name,
        k=k,
        enforce_direction=enforce_direction,
        min_weight=min_weight,
    )


def build_graphs(
    dataset: PatchEffectDataset,
    builder_name: str | None = None,
    k: int = 5,
    enforce_direction: bool = True,
) -> dict[SliceLabel, PatchInfluenceGraph]:
    """Convenience function to build graphs for all slices.

    Args:
        dataset: Dataset containing patch-effect tensors
        builder_name: Registered builder name (defaults to env/default)
        k: Number of top edges per node
        enforce_direction: Whether to enforce direction constraint

    Returns:
        Dictionary mapping slice labels to graphs
    """
    builder = create_graph_builder(
        builder_name=builder_name,
        k=k,
        enforce_direction=enforce_direction,
    )
    return builder.build_all(dataset)


def build_graphs_for_builders(
    dataset: PatchEffectDataset,
    builder_names: list[str],
    *,
    k: int = 5,
    enforce_direction: bool = True,
    min_weight: float = 0.0,
) -> dict[str, dict[SliceLabel, PatchInfluenceGraph]]:
    """Build one graph collection per registered builder name."""
    collections: dict[str, dict[SliceLabel, PatchInfluenceGraph]] = {}
    for name in builder_names:
        builder = create_graph_builder(
            builder_name=name,
            k=k,
            enforce_direction=enforce_direction,
            min_weight=min_weight,
        )
        collections[name] = builder.build_all(dataset)
    return collections
