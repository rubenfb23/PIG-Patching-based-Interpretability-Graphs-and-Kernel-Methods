"""Tests for the causal-first revision runner helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pig.causal import propose_causal_candidate_groups
from pig.patching import (
    PATCH_CACHE_SCHEMA_VERSION,
    ComponentSpec,
    PatchEffectDataset,
    PatchEffectTensor,
    build_axis_fingerprint,
)
from pig.prompts import PromptPair, SliceLabel
from scripts.run_causal_first_revision import (
    kernel_geometry_from_centroids,
    make_s10_suite,
    tail_align_dataset,
    validate_s10_suite,
)


def _prompt_pair(index: int, *, label: SliceLabel | None = None) -> PromptPair:
    return PromptPair(
        x_cln=f"clean {index}",
        x_crp=f"corrupt {index}",
        y_star="target",
        slice_label=label or SliceLabel(task="demo", corruption="a"),
        meta={},
    )


def _tensor(index: int, effects: np.ndarray) -> PatchEffectTensor:
    axis = [ComponentSpec(node_type="res")]
    return PatchEffectTensor(
        effects=effects.astype(np.float32),
        component_axis=axis,
        prompt_pair=_prompt_pair(index),
        base_score=0.0,
        clean_score=1.0,
        token_labels=[f"tok{i}" for i in range(effects.shape[1])],
        cache_schema_version=PATCH_CACHE_SCHEMA_VERSION,
        model_name="test",
        model_fingerprint="fp",
        axis_fingerprint=build_axis_fingerprint(axis),
        node_types=["res"],
        created_at_utc="2026-01-01T00:00:00+00:00",
        is_legacy_cache_entry=False,
    )


def test_candidate_groups_are_disjoint_for_res_edges():
    rng = np.random.default_rng(0)
    tensors = []
    for idx in range(12):
        effects = rng.normal(size=(3, 4, 1))
        effects[:, :, 0] += idx * 0.01
        tensors.append(_tensor(idx, effects))

    groups = propose_causal_candidate_groups(
        tensors,
        num_edges=3,
        node_types=("res",),
        seed=7,
    )

    assert set(groups) == {"top_ci", "top_pc", "random", "lowrank"}
    all_edge_ids = [
        edge.edge_id()
        for candidates in groups.values()
        for edge in candidates
    ]
    assert len(all_edge_ids) == len(set(all_edge_ids))
    assert groups["top_ci"]
    assert groups["top_pc"]


def test_tail_tokens_alignment_creates_identical_axes():
    dataset = PatchEffectDataset()
    dataset.add(_tensor(1, np.ones((2, 12, 1), dtype=np.float32)))
    dataset.add(_tensor(2, np.ones((2, 9, 1), dtype=np.float32)))

    aligned = tail_align_dataset(dataset, tail_tokens=9)

    shapes = {tensor.effects.shape for tensor in aligned}
    assert shapes == {(2, 9, 1)}
    assert all(tensor.token_labels[0] == "tok3" or len(tensor.token_labels) == 9 for tensor in aligned)


def test_s10_v1_produces_balanced_ten_slice_suite():
    pairs = make_s10_suite(2, seed=11)
    report = validate_s10_suite(pairs, expected_per_slice=2)

    assert report["num_slices"] == 10
    assert set(report["counts"].values()) == {2}


def test_kernel_geometry_refuses_too_few_slices():
    labels = [SliceLabel(task=f"t{i}", corruption="c") for i in range(4)]
    with pytest.raises(ValueError, match="at least 10 slices"):
        kernel_geometry_from_centroids(
            np.ones((4, 3), dtype=np.float64),
            labels,
            seed=7,
            null_repeats=3,
        )
