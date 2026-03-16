"""Tests for causal evaluation utilities."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pig.causal import (
    CausalEdgeCandidate,
    CausalEvalConfig,
    activation_influence_score,
    build_clean_better_filter_report,
    evaluate_causal_edges,
    load_cached_patch_effect_tensors,
    mediation_score,
    necessity_score,
    propose_causal_edge_candidates,
    restoration_fraction,
    split_discovery_evaluation_tensors,
)
from pig.model import create_model
from pig.patching import (
    PATCH_CACHE_SCHEMA_VERSION,
    ComponentSpec,
    PatchEffectTensor,
    build_axis_fingerprint,
)
from pig.prompts import PromptPair, SliceLabel, create_ioi_dataset


def _prompt_pair(index: int = 0, corruption: str = "name_swap") -> PromptPair:
    return PromptPair(
        x_cln=f"clean-{index}",
        x_crp=f"corrupt-{index}",
        y_star="target",
        slice_label=SliceLabel(task="ioi", corruption=corruption),
        meta={},
    )


def _modern_tensor(
    *,
    model_name: str,
    model_fingerprint: str,
    component_axis: list[ComponentSpec],
    index: int = 0,
    corruption: str = "name_swap",
) -> PatchEffectTensor:
    return PatchEffectTensor(
        effects=np.ones((2, 3, len(component_axis)), dtype=np.float32),
        component_axis=component_axis,
        prompt_pair=_prompt_pair(index=index, corruption=corruption),
        base_score=-1.0,
        clean_score=1.0,
        cache_schema_version=PATCH_CACHE_SCHEMA_VERSION,
        model_name=model_name,
        model_fingerprint=model_fingerprint,
        axis_fingerprint=build_axis_fingerprint(component_axis),
        node_types=["att" if c.node_type == "att" else c.node_type for c in component_axis],
        created_at_utc="2026-02-23T00:00:00+00:00",
        is_legacy_cache_entry=False,
    )


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


def test_propose_causal_edge_candidates_excludes_same_layer_att():
    """Same-layer att→att edges must be excluded: they compute in parallel."""
    rng = np.random.default_rng(0)
    # 2 layers, 3 tokens, 4 att heads → shape (n_examples, 2*3*4) = (5, 24)
    num_layers, num_tokens, num_heads = 2, 3, 4
    component_axis = [
        ComponentSpec(node_type="att", head=h) for h in range(num_heads)
    ]
    tensors = []
    for _ in range(5):
        effects = rng.random((num_layers, num_tokens, num_heads), dtype=np.float32)
        t = PatchEffectTensor(
            effects=effects,
            component_axis=component_axis,
            prompt_pair=_prompt_pair(),
            base_score=0.0,
            clean_score=1.0,
            cache_schema_version=PATCH_CACHE_SCHEMA_VERSION,
            model_name="test",
            model_fingerprint="fp",
            axis_fingerprint=build_axis_fingerprint(component_axis),
            node_types=["att"] * num_heads,
            created_at_utc="2026-01-01T00:00:00+00:00",
            is_legacy_cache_entry=False,
        )
        tensors.append(t)

    candidates = propose_causal_edge_candidates(
        tensors, num_edges=100, node_types=["att"], enforce_direction=True
    )
    for c in candidates:
        assert not (
            c.src_layer == c.dst_layer
            and c.src_node_type == "att"
            and c.dst_node_type == "att"
        ), f"Same-layer att→att candidate must not be proposed: {c}"


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
    assert "p_value_fdr_bh" in edge["level_a"]["I"]
    assert "p_value_bonferroni" in edge["level_a"]["I"]
    assert "multiple_testing" in result.run_metadata

    json_path = result.save_json(tmp_path / "causal_eval.json")
    npz_path = result.save_npz(tmp_path / "causal_eval.npz")
    assert json_path.exists()
    assert npz_path.exists()


def test_load_cached_patch_effect_tensors_rejects_model_mixing(tmp_path):
    axis_4 = [ComponentSpec(node_type="att", head=head) for head in range(4)]
    toy_tensor = _modern_tensor(
        model_name="toy_transformer",
        model_fingerprint="toy_fp",
        component_axis=axis_4,
        index=1,
    )
    gpt2_tensor = _modern_tensor(
        model_name="gpt2",
        model_fingerprint="gpt2_fp",
        component_axis=axis_4,
        index=2,
    )

    (tmp_path / "toy.json").write_text(json.dumps(toy_tensor.to_dict()), encoding="utf-8")
    (tmp_path / "gpt2.json").write_text(json.dumps(gpt2_tensor.to_dict()), encoding="utf-8")

    tensors, stats = load_cached_patch_effect_tensors(
        cache_dir=tmp_path,
        expected_model_name="toy_transformer",
        expected_model_fingerprint="toy_fp",
        expected_axis_fingerprint=build_axis_fingerprint(axis_4),
        allow_legacy_cache=False,
        return_stats=True,
    )

    assert len(tensors) == 1
    assert tensors[0].model_name == "toy_transformer"
    assert stats["discard_reasons"]["model_name_mismatch"] == 1


def test_load_cached_patch_effect_tensors_filters_axis_1_4_14(tmp_path):
    axis_1 = [ComponentSpec(node_type="res")]
    axis_4 = [ComponentSpec(node_type="att", head=head) for head in range(4)]
    axis_14 = [ComponentSpec(node_type="att", head=head) for head in range(12)] + [
        ComponentSpec(node_type="mlp"),
        ComponentSpec(node_type="res"),
    ]
    model_fingerprint = "toy_fp"

    tensors_to_write = [
        _modern_tensor(
            model_name="toy_transformer",
            model_fingerprint=model_fingerprint,
            component_axis=axis_1,
            index=1,
        ),
        _modern_tensor(
            model_name="toy_transformer",
            model_fingerprint=model_fingerprint,
            component_axis=axis_4,
            index=2,
        ),
        _modern_tensor(
            model_name="toy_transformer",
            model_fingerprint=model_fingerprint,
            component_axis=axis_14,
            index=3,
        ),
    ]
    for idx, tensor in enumerate(tensors_to_write):
        (tmp_path / f"axis_{idx}.json").write_text(
            json.dumps(tensor.to_dict()), encoding="utf-8"
        )

    tensors, stats = load_cached_patch_effect_tensors(
        cache_dir=tmp_path,
        expected_model_name="toy_transformer",
        expected_model_fingerprint=model_fingerprint,
        expected_axis_fingerprint=build_axis_fingerprint(axis_4),
        allow_legacy_cache=False,
        return_stats=True,
    )

    assert len(tensors) == 1
    assert len(tensors[0].component_axis) == 4
    assert stats["discard_reasons"]["axis_fingerprint_mismatch"] == 2


def test_load_cached_patch_effect_tensors_rejects_legacy_by_default(tmp_path):
    axis_4 = [ComponentSpec(node_type="att", head=head) for head in range(4)]
    tensor = _modern_tensor(
        model_name="toy_transformer",
        model_fingerprint="toy_fp",
        component_axis=axis_4,
        index=1,
    )
    legacy_payload = tensor.to_dict()
    for key in (
        "cache_schema_version",
        "model_name",
        "model_fingerprint",
        "axis_fingerprint",
        "node_types",
        "created_at_utc",
    ):
        legacy_payload.pop(key, None)
    (tmp_path / "legacy.json").write_text(json.dumps(legacy_payload), encoding="utf-8")

    tensors, stats = load_cached_patch_effect_tensors(
        cache_dir=tmp_path,
        expected_model_name="toy_transformer",
        expected_model_fingerprint="toy_fp",
        expected_axis_fingerprint=build_axis_fingerprint(axis_4),
        allow_legacy_cache=False,
        return_stats=True,
    )

    assert tensors == []
    assert stats["discard_reasons"]["legacy_disallowed"] == 1


def test_load_cached_patch_effect_tensors_accepts_legacy_with_flag(tmp_path):
    axis_4 = [ComponentSpec(node_type="att", head=head) for head in range(4)]
    tensor = _modern_tensor(
        model_name="toy_transformer",
        model_fingerprint="toy_fp",
        component_axis=axis_4,
        index=1,
    )
    legacy_payload = tensor.to_dict()
    for key in (
        "cache_schema_version",
        "model_name",
        "model_fingerprint",
        "axis_fingerprint",
        "node_types",
        "created_at_utc",
    ):
        legacy_payload.pop(key, None)
    (tmp_path / "legacy.json").write_text(json.dumps(legacy_payload), encoding="utf-8")

    tensors, stats = load_cached_patch_effect_tensors(
        cache_dir=tmp_path,
        expected_model_name="toy_transformer",
        expected_model_fingerprint="toy_fp",
        expected_axis_fingerprint=build_axis_fingerprint(axis_4),
        allow_legacy_cache=True,
        return_stats=True,
    )

    assert len(tensors) == 1
    assert tensors[0].is_legacy_cache_entry is True
    assert stats["accepted_legacy_tensors"] == 1


def test_build_clean_better_filter_report_includes_with_without_and_slices():
    axis_4 = [ComponentSpec(node_type="att", head=head) for head in range(4)]
    tensors = [
        _modern_tensor(
            model_name="toy_transformer",
            model_fingerprint="toy_fp",
            component_axis=axis_4,
            index=1,
            corruption="name_swap",
        ),
        _modern_tensor(
            model_name="toy_transformer",
            model_fingerprint="toy_fp",
            component_axis=axis_4,
            index=2,
            corruption="abba",
        ),
    ]
    tensors[1].clean_score = -2.0
    tensors[1].base_score = -1.0

    report = build_clean_better_filter_report(
        tensors,
        bootstrap_samples=10,
        ci_alpha=0.05,
        seed=7,
    )

    assert report["input_tensors"] == 2
    assert report["retained_with_filter"] == 1
    assert report["discarded_by_filter"] == 1
    assert report["retention_rate"] == pytest.approx(0.5)
    assert report["without_filter"]["n"] == 2
    assert report["with_filter"]["n"] == 1
    assert "ioi:name_swap" in report["without_filter"]["by_slice"]
    assert "ioi:abba" in report["without_filter"]["by_slice"]
    assert report["slice_retention"]["ioi:name_swap"]["retained_n"] == 1
    assert report["slice_retention"]["ioi:abba"]["retained_n"] == 0
    assert report["slice_retention"]["ioi:abba"]["retention_rate"] == pytest.approx(0.0)
    assert report["balance"]["before_filter"]["max_to_min_ratio"] == pytest.approx(1.0)
    assert report["balance"]["after_filter"]["max_to_min_ratio"] == pytest.approx(1.0)
    assert report["balance"]["after_filter"]["zero_count_slices"] == ["ioi:abba"]


def test_split_discovery_evaluation_tensors_stratified_and_disjoint():
    axis_4 = [ComponentSpec(node_type="att", head=head) for head in range(4)]
    tensors = [
        _modern_tensor(
            model_name="toy_transformer",
            model_fingerprint="toy_fp",
            component_axis=axis_4,
            index=1,
            corruption="name_swap",
        ),
        _modern_tensor(
            model_name="toy_transformer",
            model_fingerprint="toy_fp",
            component_axis=axis_4,
            index=2,
            corruption="name_swap",
        ),
        _modern_tensor(
            model_name="toy_transformer",
            model_fingerprint="toy_fp",
            component_axis=axis_4,
            index=3,
            corruption="abba",
        ),
        _modern_tensor(
            model_name="toy_transformer",
            model_fingerprint="toy_fp",
            component_axis=axis_4,
            index=4,
            corruption="abba",
        ),
    ]

    discovery, evaluation, stats = split_discovery_evaluation_tensors(
        tensors,
        max_total_examples=4,
        seed=42,
        discovery_fraction=0.5,
        return_stats=True,
    )

    assert len(discovery) == 2
    assert len(evaluation) == 2
    discovery_ids = {tensor.prompt_pair.x_cln for tensor in discovery}
    evaluation_ids = {tensor.prompt_pair.x_cln for tensor in evaluation}
    assert discovery_ids.isdisjoint(evaluation_ids)

    discovery_corruptions = {tensor.prompt_pair.slice_label.corruption for tensor in discovery}
    evaluation_corruptions = {tensor.prompt_pair.slice_label.corruption for tensor in evaluation}
    assert discovery_corruptions == {"name_swap", "abba"}
    assert evaluation_corruptions == {"name_swap", "abba"}
    assert stats["selected_discovery_count"] == 2
    assert stats["selected_evaluation_count"] == 2


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
