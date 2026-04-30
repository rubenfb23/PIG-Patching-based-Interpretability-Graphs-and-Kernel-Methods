"""Tests for the lightweight GNN graph baseline."""

import pytest

from pig.gnn import GNNTrainingConfig, cross_validate_gnn_baseline
from pig.graph import Edge, Node, PatchInfluenceGraph
from pig.prompts import SliceLabel


def _make_graph(corruption: str, weight: float) -> PatchInfluenceGraph:
    nodes = [Node(layer=0, token=0), Node(layer=0, token=1), Node(layer=1, token=0)]
    return PatchInfluenceGraph(
        nodes=nodes,
        edges=[
            Edge(src=0, dst=1, weight=weight),
            Edge(src=1, dst=2, weight=weight / 2),
        ],
        slice_label=SliceLabel(task="ioi", corruption=corruption),
        num_layers=2,
        num_tokens=2,
    )


def test_cross_validate_gnn_baseline_returns_fold_metrics():
    graphs_with_labels = []
    for idx in range(4):
        label = SliceLabel(task="ioi", corruption="name_swap")
        graphs_with_labels.append((_make_graph("name_swap", 0.7 + idx * 0.01), label))
    for idx in range(4):
        label = SliceLabel(task="ioi", corruption="abba")
        graphs_with_labels.append((_make_graph("abba", -0.7 - idx * 0.01), label))

    result = cross_validate_gnn_baseline(
        graphs_with_labels,
        config=GNNTrainingConfig(
            hidden_dim=8,
            epochs=5,
            random_state=7,
            device="cpu",
        ),
        cv=2,
    )

    assert result.num_graphs == 8
    assert result.cv_folds == 2
    assert len(result.fold_accuracies) == 2
    assert 0.0 <= result.accuracy_mean <= 1.0
    assert all(0.0 <= accuracy <= 1.0 for accuracy in result.fold_accuracies)


def test_cross_validate_gnn_baseline_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        cross_validate_gnn_baseline([])


def test_cross_validate_gnn_baseline_requires_two_samples_per_class():
    label_a = SliceLabel(task="ioi", corruption="name_swap")
    label_b = SliceLabel(task="ioi", corruption="abba")

    with pytest.raises(ValueError, match="two samples per class"):
        cross_validate_gnn_baseline(
            [
                (_make_graph("name_swap", 0.5), label_a),
                (_make_graph("abba", -0.5), label_b),
            ]
        )
