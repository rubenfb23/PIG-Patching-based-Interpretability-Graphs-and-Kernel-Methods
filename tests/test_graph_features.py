"""Tests for fixed-layout graph features and null controls."""

import numpy as np
import pytest

from pig.graph import Edge, Node, PatchInfluenceGraph
from pig.graph_features import (
    apply_null_control_to_graphs,
    compute_fixed_layout_features_from_list,
    shuffle_graph_edges,
    shuffle_graph_weights,
)
from pig.prompts import SliceLabel


@pytest.fixture
def sample_graph():
    nodes = [Node(layer=0, token=0), Node(layer=0, token=1), Node(layer=1, token=0)]
    edges = [
        Edge(src=0, dst=1, weight=0.5),
        Edge(src=0, dst=2, weight=-0.25),
    ]
    return PatchInfluenceGraph(
        nodes=nodes,
        edges=edges,
        slice_label=SliceLabel(task="ioi", corruption="name_swap"),
        num_layers=2,
        num_tokens=2,
    )


def test_fixed_layout_feature_matrix_preserves_weights(sample_graph):
    label = sample_graph.slice_label
    feature_matrix = compute_fixed_layout_features_from_list([(sample_graph, label)])
    X = feature_matrix.to_matrix()

    assert X.shape == (1, 6)
    assert feature_matrix.num_graphs == 1
    assert feature_matrix.num_features == 6
    assert np.count_nonzero(X) == 2
    assert np.sum(X) == pytest.approx(0.25)


def test_fixed_layout_feature_order_is_deterministic(sample_graph):
    label = sample_graph.slice_label
    first = compute_fixed_layout_features_from_list([(sample_graph, label)])
    second = compute_fixed_layout_features_from_list([(sample_graph, label)])

    assert first.feature_names == second.feature_names
    np.testing.assert_array_equal(first.to_matrix(), second.to_matrix())


def test_fixed_layout_handles_empty_graph(sample_graph):
    empty_graph = PatchInfluenceGraph(
        nodes=sample_graph.nodes,
        edges=[],
        slice_label=sample_graph.slice_label,
        num_layers=sample_graph.num_layers,
        num_tokens=sample_graph.num_tokens,
    )

    feature_matrix = compute_fixed_layout_features_from_list(
        [(empty_graph, empty_graph.slice_label)]
    )

    assert feature_matrix.to_matrix().shape == (1, 6)
    assert np.count_nonzero(feature_matrix.to_matrix()) == 0


def test_weight_shuffle_preserves_topology_and_weight_multiset(sample_graph):
    rng = np.random.default_rng(7)
    shuffled = shuffle_graph_weights(sample_graph, rng)

    assert [(e.src, e.dst) for e in shuffled.edges] == [
        (e.src, e.dst) for e in sample_graph.edges
    ]
    assert sorted(e.weight for e in shuffled.edges) == pytest.approx(
        sorted(e.weight for e in sample_graph.edges)
    )
    assert shuffled.metadata["null_control"] == "weight_shuffle"


def test_edge_shuffle_preserves_direction_and_weight_multiset(sample_graph):
    rng = np.random.default_rng(7)
    shuffled = shuffle_graph_edges(sample_graph, rng)

    assert len(shuffled.edges) == len(sample_graph.edges)
    assert sorted(e.weight for e in shuffled.edges) == pytest.approx(
        sorted(e.weight for e in sample_graph.edges)
    )
    for edge in shuffled.edges:
        assert shuffled.nodes[edge.src] < shuffled.nodes[edge.dst]
    assert shuffled.metadata["null_control"] == "edge_shuffle"


def test_apply_null_control_keeps_labels(sample_graph):
    label = sample_graph.slice_label
    transformed = apply_null_control_to_graphs(
        [(sample_graph, label)],
        "weight_shuffle",
        seed=7,
    )

    assert transformed[0][1] == label
    assert transformed[0][0].metadata["null_control"] == "weight_shuffle"
