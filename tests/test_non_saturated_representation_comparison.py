"""Tests for the paired non-saturated representation experiment."""

from __future__ import annotations

import copy

import numpy as np

from pig.graph import Edge, Node, PatchInfluenceGraph
from pig.graph_features import compute_directed_motif_features_from_list
from pig.patching import ComponentSpec, PatchEffectDataset, PatchEffectTensor
from pig.prompts import PromptPair, SliceLabel
from scripts.run_non_saturated_representation_comparison import (
    ALL_REPRESENTATIONS,
    LABELS,
    NAME_SPLITS,
    TEMPLATE_INDICES,
    FeatureBundle,
    apply_context_condition,
    assert_paired_bags,
    build_rebuttal_markdown,
    fit_pca_bundle,
    generate_base_pairs,
    load_seed_checkpoint,
    make_bags,
    paired_bootstrap_delta,
    summarize_rows,
    symmetric_noise_labels,
    validate_pair_suite,
)


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        _ = add_special_tokens
        return list(range(len(text.split())))


def _tensor(index: int, label: SliceLabel, value: float) -> PatchEffectTensor:
    pair = PromptPair(
        x_cln=f"clean {index}",
        x_crp=f"corrupt {index}",
        y_star="John",
        slice_label=label,
        meta={"y_distractor": "Mary"},
    )
    return PatchEffectTensor(
        effects=np.full((2, 2, 1), value, dtype=np.float32),
        component_axis=[ComponentSpec("res")],
        prompt_pair=pair,
        base_score=-1.0,
        clean_score=1.0,
    )


def test_generation_is_balanced_deterministic_and_surface_matched():
    first = generate_base_pairs(split="train", n_per_class=6, seed=7)
    second = generate_base_pairs(split="train", n_per_class=6, seed=7)
    assert [pair.to_dict() for pair in first] == [pair.to_dict() for pair in second]

    contextualized = apply_context_condition(
        first,
        split="train",
        tier=2,
        seed=99,
    )
    report = validate_pair_suite(
        contextualized,
        WordTokenizer(),
        expected_per_class=6,
    )
    assert report["counts"] == {label: 6 for label in LABELS}
    assert all(values == [(2, 1)] for values in report["surface_counts"].values())


def test_prompt_family_splits_are_disjoint():
    name_sets = [set(values) for values in NAME_SPLITS.values()]
    assert not (name_sets[0] & name_sets[1])
    assert not (name_sets[0] & name_sets[2])
    assert not (name_sets[1] & name_sets[2])
    template_sets = [set(values) for values in TEMPLATE_INDICES.values()]
    assert not (template_sets[0] & template_sets[1])
    assert not (template_sets[0] & template_sets[2])
    assert not (template_sets[1] & template_sets[2])


def test_bag_membership_is_paired():
    label_a = SliceLabel("ioi", "abba")
    label_b = SliceLabel("ioi", "second_subject_swap")
    first_dataset = PatchEffectDataset(
        [_tensor(i, label_a, i) for i in range(4)]
        + [_tensor(i, label_b, i + 10) for i in range(4)]
    )
    second_dataset = PatchEffectDataset(
        [_tensor(i, label_a, i + 100) for i in range(4)]
        + [_tensor(i, label_b, i + 110) for i in range(4)]
    )
    first_bags = make_bags(first_dataset, split="test", bag_size=2)
    second_bags = make_bags(second_dataset, split="test", bag_size=2)
    assert_paired_bags(first_bags, second_bags)
    assert [bag.bag_id for bag in first_bags] == [bag.bag_id for bag in second_bags]


def test_symmetric_noise_is_exact_balanced_and_deterministic():
    labels = ["a"] * 20 + ["b"] * 20
    first = symmetric_noise_labels(labels, rate=0.2, seed=7)
    second = symmetric_noise_labels(labels, rate=0.2, seed=7)
    assert first == second
    assert sum(a != b for a, b in zip(labels, first, strict=True)) == 8
    assert first.count("a") == first.count("b") == 20


def test_pca_is_fit_on_train_only():
    train = FeatureBundle(
        X=np.arange(60, dtype=np.float32).reshape(10, 6),
        y=["a"] * 5 + ["b"] * 5,
        ids=[str(index) for index in range(10)],
        feature_names=[f"x{index}" for index in range(6)],
    )
    evaluation = FeatureBundle(
        X=np.ones((4, 6), dtype=np.float32),
        y=["a", "a", "b", "b"],
        ids=[f"e{index}" for index in range(4)],
        feature_names=train.feature_names,
    )
    shifted = copy.deepcopy(evaluation)
    shifted.X *= 1_000_000
    first_train, _ = fit_pca_bundle(train, [evaluation], target_dim=3, seed=7)
    second_train, _ = fit_pca_bundle(train, [shifted], target_dim=3, seed=7)
    np.testing.assert_allclose(first_train.X, second_train.X)


def test_directed_motifs_are_invariant_to_node_permutation():
    label = SliceLabel("ioi", "abba")
    nodes = [Node(layer=0, token=index) for index in range(4)]
    graph = PatchInfluenceGraph(
        nodes=nodes,
        edges=[
            Edge(0, 1, 0.5),
            Edge(1, 2, -0.2),
            Edge(2, 0, 0.7),
            Edge(2, 3, 0.3),
        ],
        slice_label=label,
        num_layers=1,
        num_tokens=4,
    )
    permutation = [2, 0, 3, 1]
    inverse = {old: new for new, old in enumerate(permutation)}
    permuted = PatchInfluenceGraph(
        nodes=[nodes[index] for index in permutation],
        edges=[
            Edge(inverse[edge.src], inverse[edge.dst], edge.weight)
            for edge in graph.edges
        ],
        slice_label=label,
        num_layers=1,
        num_tokens=4,
    )
    features = compute_directed_motif_features_from_list(
        [(graph, label), (permuted, label)]
    ).to_matrix()
    np.testing.assert_allclose(features[0], features[1])
    assert features.shape[1] == 36
    assert np.sum(features[0, :16]) == 1.0


def test_checkpoint_resume_requires_matching_signature(tmp_path):
    path = tmp_path / "seed.json"
    path.write_text('{"config_signature": "abc", "seed": 7}', encoding="utf-8")
    assert load_seed_checkpoint(path, config_signature="abc")["seed"] == 7
    try:
        load_seed_checkpoint(path, config_signature="different")
    except ValueError as error:
        assert "configuration mismatch" in str(error)
    else:
        raise AssertionError("Expected mismatched checkpoint to fail")


def test_paired_bootstrap_targets_mean_seed_delta():
    labels = ["ioi:abba"] * 4 + ["ioi:second_subject_swap"] * 4
    raw_predictions = ["ioi:abba"] * 8
    seed_payloads = []
    for seed in (7, 42):
        seed_payloads.append(
            {
                "predictions": {
                    "selected_raw": {
                        "0.0": {
                            "test_shift": {
                                "y_true": labels,
                                "predictions": raw_predictions,
                            }
                        }
                    },
                    "selected_graph": {
                        "0.0": {
                            "test_shift": {
                                "y_true": labels,
                                "predictions": labels,
                            }
                        }
                    },
                }
            }
        )
    result = paired_bootstrap_delta(
        seed_payloads,
        metric="macro_f1",
        noise_rate=0.0,
        condition="test_shift",
        samples=200,
        seed=2026,
    )
    assert result["mean_seed_delta"] == result["ci_low"]
    assert result["mean_seed_delta"] == result["ci_high"]


def test_summary_and_markdown_table_are_complete():
    rows = []
    representations = [*ALL_REPRESENTATIONS, "selected_raw", "selected_graph"]
    for seed in (7, 42):
        for representation in representations:
            for noise_rate in (0.0, 0.2):
                for condition in ("test_clean", "test_shift"):
                    rows.append(
                        {
                            "seed": seed,
                            "representation": representation,
                            "selected_variant": "",
                            "noise_rate": noise_rate,
                            "condition": condition,
                            "accuracy": 0.7,
                            "macro_f1": 0.68,
                            "num_features": 36,
                            "num_train": 20,
                            "num_test": 10,
                        }
                    )
    summary = summarize_rows(rows)
    y_true = ["ioi:abba", "ioi:second_subject_swap"]
    predictions = {
        selected: {
            "0.0": {
                condition: {
                    "y_true": y_true,
                    "predictions": y_true,
                    "ids": ["a", "b"],
                }
                for condition in ("test_clean", "test_shift")
            }
        }
        for selected in ("selected_raw", "selected_graph")
    }
    seed_payloads = [
        {
            "seed": seed,
            "rows": [row for row in rows if row["seed"] == seed],
            "predictions": predictions,
        }
        for seed in (7, 42)
    ]
    gate = {
        "decision": "NO-GO",
        "checks": {"example": False},
        "shift_accuracy_delta": {
            "mean_seed_delta": 0.0,
            "ci_low": -0.1,
            "ci_high": 0.1,
        },
        "shift_macro_f1_delta": {
            "mean_seed_delta": 0.0,
            "ci_low": -0.1,
            "ci_high": 0.1,
        },
    }
    markdown = build_rebuttal_markdown(
        summary,
        seed_payloads,
        gate,
        selected_tier=2,
        bag_size=12,
    )
    assert "**Preregistered decision: NO-GO**" in markdown
    assert "Graph directed motifs" in markdown
    assert "20% noise + shift macro-F1" in markdown
