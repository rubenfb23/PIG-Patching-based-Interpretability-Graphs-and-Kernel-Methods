"""Tests for exhaustive direct-influence graph construction and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from pig.direct_influence import (
    FullGraphConfig,
    HeadNode,
    PromptGraphScores,
    aggregate_full_graph_scores,
    enumerate_forward_edges,
    enumerate_head_nodes,
    evaluate_prompt_full_graph,
    graph_alignment_metrics,
)
from pig.model import create_model
from pig.prompts import create_ioi_dataset


def test_full_head_universe_has_all_cross_layer_edges():
    nodes = enumerate_head_nodes(n_layers=3, n_heads=2)
    edges = enumerate_forward_edges(nodes)

    assert len(nodes) == 6
    assert len(edges) == 12
    assert len({edge.edge_id() for edge in edges}) == len(edges)
    assert all(edge.source.layer < edge.target.layer for edge in edges)


def test_graph_alignment_metrics_controlled():
    predicted = np.asarray([True, True, False, False])
    reference = np.asarray([True, False, True, False])
    result = graph_alignment_metrics(
        predicted,
        reference,
        predicted_weights=np.asarray([2.0, 1.0, 0.0, 0.0]),
        reference_weights=np.asarray([2.0, 0.0, 1.0, 0.0]),
    )

    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["jaccard"] == pytest.approx(1.0 / 3.0)
    assert result["graph_edit_distance"] == 2
    assert 0.0 < result["kernel_alignment"] < 1.0


def test_aggregate_full_graph_scores_builds_requested_table():
    nodes = [HeadNode(0, 0), HeadNode(1, 0), HeadNode(2, 0)]
    edges = enumerate_forward_edges(nodes)
    edge_ids = [edge.edge_id() for edge in edges]
    first = PromptGraphScores(
        edge_ids=edge_ids,
        direct_influence=np.asarray([0.8, 0.4, 0.1]),
        causal_intervention=np.asarray([0.7, 0.2, -0.1]),
        path_patching=np.asarray([0.6, -0.1, 0.3]),
        metadata={},
    )
    second = PromptGraphScores(
        edge_ids=edge_ids,
        direct_influence=np.asarray([0.9, 0.5, 0.2]),
        causal_intervention=np.asarray([0.8, 0.3, -0.2]),
        path_patching=np.asarray([0.7, -0.2, 0.4]),
        metadata={},
    )

    result = aggregate_full_graph_scores(
        [first, second],
        nodes,
        edges,
        FullGraphConfig(bootstrap_samples=20, edge_budget=1, seed=7),
    )

    assert result["metadata"]["full_edge_universe"] is True
    assert result["metadata"]["num_edges"] == 3
    assert len(result["summary_table"]) == 4
    assert {(row["method"], row["reference"]) for row in result["summary_table"]} >= {
        ("direct_influence", "causal_intervention"),
        ("path_patching", "direct_influence"),
    }
    assert len(result["edge_rows"]) == 3


def test_evaluate_prompt_full_graph_toy_smoke():
    model = create_model(model_name="toy_transformer", device="cpu")
    prompt = create_ioi_dataset(1, corruption="name_swap", seed=3)[0]
    nodes = enumerate_head_nodes(
        model.n_layers,
        model.n_heads,
        heads=[0, 1],
    )
    edges = enumerate_forward_edges(nodes)

    scores = evaluate_prompt_full_graph(
        model,
        prompt,
        nodes,
        edges,
        show_progress=False,
    )

    assert len(edges) == 4
    assert scores.edge_ids == [edge.edge_id() for edge in edges]
    assert np.all(np.isfinite(scores.direct_influence))
    assert np.all(np.isfinite(scores.causal_intervention))
    assert np.all(np.isfinite(scores.path_patching))
