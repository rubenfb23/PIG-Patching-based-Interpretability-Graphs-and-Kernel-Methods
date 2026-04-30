"""Fixed-layout graph features and null graph controls.

These utilities support auxiliary representation checks around WL features.
They intentionally keep the causal graph construction untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pig.graph import Edge, Node, PatchInfluenceGraph
from pig.prompts import SliceLabel

NullControl = Literal["edge_shuffle", "weight_shuffle"]


def _node_key(node: Node) -> str:
    head = "none" if node.head is None else str(node.head)
    return f"L{node.layer}:T{node.token}:{node.node_type}:H{head}"


def _edge_feature_key(src: Node, dst: Node) -> str:
    return f"edge|{_node_key(src)}->{_node_key(dst)}"


@dataclass
class FixedLayoutFeatureMatrix:
    """Dense fixed-layout edge-weight features for graph classifiers."""

    matrix: NDArray[np.float32]
    feature_names: list[str]
    slice_labels: list[SliceLabel]

    @property
    def num_graphs(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.matrix.shape[1])

    def to_matrix(self) -> NDArray[np.float32]:
        return self.matrix


def compute_fixed_layout_features_from_list(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
) -> FixedLayoutFeatureMatrix:
    """Vectorize graphs using explicit directed edge slots.

    Unlike WL, this baseline uses the fixed transformer layout directly:
    every possible observed ``source node -> destination node`` slot becomes
    one feature, and the value is the edge weight or zero.
    """
    graph_list = [graph for graph, _ in graphs_with_labels]
    slice_labels = [label for _, label in graphs_with_labels]
    feature_names = _build_edge_vocabulary(graph_list)
    feature_index = {name: index for index, name in enumerate(feature_names)}
    matrix = np.zeros((len(graph_list), len(feature_names)), dtype=np.float32)

    for row_idx, graph in enumerate(graph_list):
        for edge in graph.edges:
            src = graph.nodes[edge.src]
            dst = graph.nodes[edge.dst]
            key = _edge_feature_key(src, dst)
            col_idx = feature_index.get(key)
            if col_idx is not None:
                matrix[row_idx, col_idx] += np.float32(edge.weight)

    return FixedLayoutFeatureMatrix(
        matrix=matrix,
        feature_names=feature_names,
        slice_labels=slice_labels,
    )


def apply_null_control_to_graphs(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    control: NullControl,
    *,
    seed: int = 42,
) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
    """Apply a graph null control while preserving graph labels."""
    rng = np.random.default_rng(seed)
    transformed: list[tuple[PatchInfluenceGraph, SliceLabel]] = []
    for graph, label in graphs_with_labels:
        if control == "edge_shuffle":
            null_graph = shuffle_graph_edges(graph, rng)
        elif control == "weight_shuffle":
            null_graph = shuffle_graph_weights(graph, rng)
        else:
            raise ValueError(f"Unknown null control: {control}")
        transformed.append((null_graph, label))
    return transformed


def shuffle_graph_weights(
    graph: PatchInfluenceGraph,
    rng: np.random.Generator,
) -> PatchInfluenceGraph:
    """Shuffle edge weights across the existing topology."""
    weights = [edge.weight for edge in graph.edges]
    if weights:
        shuffled = rng.permutation(np.array(weights, dtype=np.float32)).tolist()
    else:
        shuffled = []
    edges = [
        Edge(src=edge.src, dst=edge.dst, weight=float(shuffled[index]))
        for index, edge in enumerate(graph.edges)
    ]
    return _clone_graph(
        graph,
        edges,
        null_control="weight_shuffle",
    )


def shuffle_graph_edges(
    graph: PatchInfluenceGraph,
    rng: np.random.Generator,
) -> PatchInfluenceGraph:
    """Shuffle edge endpoints while preserving the edge-weight multiset."""
    if not graph.edges:
        return _clone_graph(graph, [], null_control="edge_shuffle")

    preserve_direction = _all_edges_follow_direction(graph)
    candidate_pairs = _candidate_edge_pairs(graph.nodes, preserve_direction)
    if not candidate_pairs:
        return _clone_graph(graph, [], null_control="edge_shuffle")

    replace = len(graph.edges) > len(candidate_pairs)
    chosen_indices = rng.choice(
        len(candidate_pairs),
        size=len(graph.edges),
        replace=replace,
    )
    shuffled_weights = rng.permutation(
        np.array([edge.weight for edge in graph.edges], dtype=np.float32)
    )
    edges = []
    for edge_idx, pair_idx in enumerate(chosen_indices):
        src, dst = candidate_pairs[int(pair_idx)]
        edges.append(Edge(src=src, dst=dst, weight=float(shuffled_weights[edge_idx])))

    return _clone_graph(
        graph,
        edges,
        null_control="edge_shuffle",
        preserve_direction=preserve_direction,
    )


def _build_edge_vocabulary(graphs: list[PatchInfluenceGraph]) -> list[str]:
    features: set[str] = set()
    for graph in graphs:
        for src in graph.nodes:
            for dst in graph.nodes:
                if src == dst:
                    continue
                features.add(_edge_feature_key(src, dst))
    return sorted(features)


def _all_edges_follow_direction(graph: PatchInfluenceGraph) -> bool:
    return all(graph.nodes[edge.src] < graph.nodes[edge.dst] for edge in graph.edges)


def _candidate_edge_pairs(
    nodes: list[Node],
    preserve_direction: bool,
) -> list[tuple[int, int]]:
    pairs = []
    for src_idx, src in enumerate(nodes):
        for dst_idx, dst in enumerate(nodes):
            if src_idx == dst_idx:
                continue
            if preserve_direction and not (src < dst):
                continue
            pairs.append((src_idx, dst_idx))
    return pairs


def _clone_graph(
    graph: PatchInfluenceGraph,
    edges: list[Edge],
    **metadata: object,
) -> PatchInfluenceGraph:
    graph_metadata = dict(graph.metadata)
    graph_metadata.update(metadata)
    return PatchInfluenceGraph(
        nodes=list(graph.nodes),
        edges=edges,
        slice_label=graph.slice_label,
        num_layers=graph.num_layers,
        num_tokens=graph.num_tokens,
        metadata=graph_metadata,
    )
