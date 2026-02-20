"""Tests for causal evaluation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from pig.causal import (
    CausalEdgeCandidate,
    CausalEvalConfig,
    activation_influence_score,
    evaluate_causal_edges,
    mediation_score,
    necessity_score,
    restoration_fraction,
)
from pig.model import create_model
from pig.prompts import create_ioi_dataset


def test_activation_influence_score_controlled():
    base = np.array([0.0, 0.0], dtype=np.float64)
    clean = np.array([2.0, 0.0], dtype=np.float64)
    patched = np.array([1.0, 0.0], dtype=np.float64)

    score = activation_influence_score(patched, base, clean, eps=1e-9)
    assert score == pytest.approx(0.5)


def test_restoration_mediation_necessity_formulas():
    r_u = restoration_fraction(12.0, base_score=10.0, clean_score=14.0, eps=1e-9)
    r_v = restoration_fraction(11.0, base_score=10.0, clean_score=14.0, eps=1e-9)
    r_uv = restoration_fraction(13.0, base_score=10.0, clean_score=14.0, eps=1e-9)
    r_u_clamp = restoration_fraction(10.5, base_score=10.0, clean_score=14.0, eps=1e-9)

    assert r_u == pytest.approx(0.5)
    assert r_v == pytest.approx(0.25)
    assert r_uv == pytest.approx(0.75)
    assert mediation_score(r_uv, r_v) == pytest.approx(0.5)
    assert necessity_score(r_u, r_u_clamp) == pytest.approx(0.375)


def test_evaluate_causal_edges_toy_smoke(tmp_path):
    model = create_model(model_name="toy_transformer", device="cpu")
    pairs = create_ioi_dataset(n_examples=2, corruption="name_swap", seed=7)
    candidates = [
        CausalEdgeCandidate(
            src_layer=0,
            src_token=0,
            src_node_type="att",
            src_head=0,
            dst_layer=1,
            dst_token=1,
            dst_node_type="att",
            dst_head=0,
        )
    ]
    config = CausalEvalConfig(
        eps=1e-6,
        bootstrap_samples=10,
        permutation_samples=10,
        seed=123,
    )

    result = evaluate_causal_edges(model, pairs, candidates, config)
    assert len(result.edges) == 1

    edge = result.edges[0]
    assert np.isfinite(edge["level_a"]["I"]["mean"])
    assert np.isfinite(edge["level_b"]["M"]["mean"])
    assert np.isfinite(edge["level_c"]["necessity"]["mean"])

    json_path = result.save_json(tmp_path / "causal_eval.json")
    npz_path = result.save_npz(tmp_path / "causal_eval.npz")
    assert json_path.exists()
    assert npz_path.exists()


@pytest.mark.slow
def test_evaluate_causal_edges_gpt2_integration(tmp_path):
    model = create_model(model_name="gpt2", device="cpu")
    pairs = create_ioi_dataset(n_examples=2, corruption="name_swap", seed=11)
    candidates = [
        CausalEdgeCandidate(
            src_layer=0,
            src_token=0,
            src_node_type="att",
            src_head=0,
            dst_layer=1,
            dst_token=1,
            dst_node_type="att",
            dst_head=0,
        )
    ]
    config = CausalEvalConfig(
        eps=1e-6,
        bootstrap_samples=5,
        permutation_samples=5,
        seed=11,
    )

    result = evaluate_causal_edges(model, pairs, candidates, config)
    json_path = result.save_json(tmp_path / "causal_eval.json")
    npz_path = result.save_npz(tmp_path / "causal_eval.npz")

    assert json_path.exists()
    assert npz_path.exists()
    arrays = np.load(npz_path)
    assert "I_mean" in arrays
    assert arrays["I_mean"].shape[0] == 1
