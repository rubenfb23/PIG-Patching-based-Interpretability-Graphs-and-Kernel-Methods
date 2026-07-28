#!/usr/bin/env python3
"""Leakage-safe raw patch-effect versus graph-representation comparison.

The unit of classification is a non-overlapping bag of prompt pairs. Raw and
graph representations are derived from exactly the same tensors in each bag.
The primary task distinguishes two surface-balanced IOI corruption slices under
held-out names/templates, unseen context distractors, and training-label noise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE",
    str(ROOT / ".cache" / "huggingface" / "hub2"),
)

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from pig.graph import PatchInfluenceGraph, create_graph_builder  # noqa: E402
from pig.graph_features import compute_directed_motif_features_from_list  # noqa: E402
from pig.model import create_model  # noqa: E402
from pig.patching import (  # noqa: E402
    PatchEffectDataset,
    PatchEffectTensor,
    compute_patch_effects,
)
from pig.prompts import (  # noqa: E402
    ABBACorruption,
    IOIGenerator,
    PromptPair,
    SecondSubjectSwapCorruption,
    SliceLabel,
    get_prompt_pair_distractor,
)


FINAL_SEEDS = (7, 42, 123, 256, 512)
LABELS = ("ioi:abba", "ioi:second_subject_swap")
TRAIN_NAMES = (
    "John",
    "Mary",
    "Alice",
    "Bob",
    "James",
    "Sarah",
    "Michael",
    "Lisa",
    "David",
    "Jennifer",
    "Robert",
    "Emily",
)
VALIDATION_NAMES = ("William", "Jessica", "Thomas")
TEST_NAMES = ("Daniel", "Matthew", "Christopher", "Andrew")
NAME_SPLITS = {
    "train": TRAIN_NAMES,
    "validation": VALIDATION_NAMES,
    "test": TEST_NAMES,
}
TEMPLATE_INDICES = {
    "train": (0, 1, 2),
    "validation": (3,),
    "test": (4, 5),
}
CONTEXT_POOLS = {
    "train": (
        "Earlier, the weather remained calm.",
        "Nearby, a teacher closed the window.",
        "Before lunch, a neighbor moved a chair.",
        "In another room, a nurse read a note.",
        "Outside, a driver waited beside the gate.",
        "That morning, a painter cleaned a brush.",
    ),
    "validation": (
        "At sunrise, a baker arranged the trays.",
        "Across town, a pilot checked the map.",
        "During the break, a singer opened a case.",
        "In the hallway, a guard carried a folder.",
        "Later, a farmer repaired the fence.",
        "By the stairs, a doctor found a pen.",
    ),
    "test": (
        "Before the meeting, a sailor folded a chart.",
        "In the courtyard, a chef carried a basket.",
        "At the station, a dancer checked a ticket.",
        "Behind the building, a clerk moved a parcel.",
        "Near the library, a coach opened a cabinet.",
        "After the storm, a tailor counted the boxes.",
    ),
}
COMMON_CONTEXT_ANCHOR = "The room was quiet."
RAW_REPRESENTATIONS = ("raw_mean", "raw_mean_std", "raw_pca_motif")
GRAPH_REPRESENTATIONS = (
    "graph_edge_slot_signed",
    "graph_edge_slot_weighted",
    "graph_directed_motifs",
)
ALL_REPRESENTATIONS = (
    *RAW_REPRESENTATIONS,
    *GRAPH_REPRESENTATIONS,
    "surface_cue_control",
)
NOISE_RATES = (0.0, 0.1, 0.2, 0.3)


@dataclass
class Bag:
    bag_id: str
    label: str
    tensors: list[PatchEffectTensor]


@dataclass
class FeatureBundle:
    X: np.ndarray
    y: list[str]
    ids: list[str]
    feature_names: list[str]


class ToyTokenizerAdapter:
    """Minimal encode/decode adapter for the in-repo toy model tokenizer."""

    def __init__(self, model: Any):
        self.model = model

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        _ = add_special_tokens
        return list(self.model._split_tokens(text))

    def decode(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def stable_hash(payload: Any, length: int = 20) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _corruption_objects() -> tuple[tuple[str, object], ...]:
    return (
        ("abba", ABBACorruption()),
        ("second_subject_swap", SecondSubjectSwapCorruption()),
    )


def generate_base_pairs(
    *,
    split: str,
    n_per_class: int,
    seed: int,
) -> list[PromptPair]:
    """Generate balanced base prompt pairs from split-disjoint names/templates."""
    if split not in NAME_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    names = list(NAME_SPLITS[split])
    templates = [IOIGenerator.TEMPLATES[index] for index in TEMPLATE_INDICES[split]]
    pairs: list[PromptPair] = []
    for offset, (_, corruption) in enumerate(_corruption_objects()):
        generator = IOIGenerator(
            corruption=corruption,
            seed=seed + offset * 100_000,
            names=names,
            templates=templates,
        )
        pairs.extend(generator.generate_batch(n_per_class))
    return pairs


def _insert_context(prompt: str, prefix: Sequence[str], middle: Sequence[str]) -> str:
    first, separator, remainder = prompt.partition(". ")
    prefix_text = " ".join(prefix)
    middle_text = " ".join(middle)
    body = prompt
    if separator:
        body = f"{first}. {middle_text + ' ' if middle_text else ''}{remainder}"
    if prefix_text:
        body = f"{prefix_text} {body}"
    return body


def apply_context_condition(
    pairs: Sequence[PromptPair],
    *,
    split: str,
    tier: int,
    seed: int,
) -> list[PromptPair]:
    """Apply deterministic label-neutral context perturbations.

    Tier 0 is clean. Tier -1 is the training mixture with zero or one prefix
    clause. Tiers 1--3 add that many unseen clauses, alternating prefix and
    between-sentence placement.
    """
    if tier not in {-1, 0, 1, 2, 3}:
        raise ValueError("tier must be one of -1, 0, 1, 2, 3")
    pool = CONTEXT_POOLS[split]
    rng = np.random.default_rng(seed)
    transformed: list[PromptPair] = []
    for pair_index, pair in enumerate(pairs):
        clause_count = int(rng.integers(0, 2)) if tier == -1 else tier
        if clause_count:
            order = rng.permutation(len(pool))
            clauses = [pool[int(index)] for index in order[:clause_count]]
        else:
            clauses = []
        prefix = [COMMON_CONTEXT_ANCHOR, *clauses[::2]]
        middle = clauses[1::2]
        meta = dict(pair.meta)
        meta.update(
            {
                "context_split": split,
                "context_tier": tier,
                "context_clauses": clauses,
                "base_pair_index": pair_index,
            }
        )
        transformed.append(
            replace(
                pair,
                x_cln=_insert_context(pair.x_cln, prefix, middle),
                x_crp=_insert_context(pair.x_crp, prefix, middle),
                meta=meta,
            )
        )
    return transformed


def tail_truncate_pair(
    pair: PromptPair,
    tokenizer: Any,
    *,
    tail_tokens: int,
) -> PromptPair:
    """Truncate clean/corrupted prompts to an answer-aligned token window."""
    clean_ids = tokenizer.encode(pair.x_cln, add_special_tokens=False)
    corrupted_ids = tokenizer.encode(pair.x_crp, add_special_tokens=False)
    if len(clean_ids) < tail_tokens or len(corrupted_ids) < tail_tokens:
        raise ValueError(
            f"Prompt has fewer than tail_tokens={tail_tokens}: "
            f"clean={len(clean_ids)}, corrupted={len(corrupted_ids)}"
        )
    clean_tail = clean_ids[-tail_tokens:]
    corrupted_tail = corrupted_ids[-tail_tokens:]
    clean_text = tokenizer.decode(clean_tail)
    corrupted_text = tokenizer.decode(corrupted_tail)
    if tokenizer.encode(clean_text, add_special_tokens=False) != clean_tail:
        raise ValueError("Clean prompt tail failed tokenizer round trip")
    if tokenizer.encode(corrupted_text, add_special_tokens=False) != corrupted_tail:
        raise ValueError("Corrupted prompt tail failed tokenizer round trip")
    meta = dict(pair.meta)
    meta["pretruncate_clean"] = pair.x_cln
    meta["pretruncate_corrupted"] = pair.x_crp
    meta["tail_tokens"] = tail_tokens
    return replace(pair, x_cln=clean_text, x_crp=corrupted_text, meta=meta)


def prepare_condition_pairs(
    base_pairs: Sequence[PromptPair],
    *,
    split: str,
    tier: int,
    seed: int,
    tokenizer: Any,
    tail_tokens: int,
) -> list[PromptPair]:
    contextualized = apply_context_condition(
        base_pairs,
        split=split,
        tier=tier,
        seed=seed,
    )
    return [
        tail_truncate_pair(pair, tokenizer, tail_tokens=tail_tokens)
        for pair in contextualized
    ]


def validate_pair_suite(
    pairs: Sequence[PromptPair],
    tokenizer: Any,
    *,
    expected_per_class: int,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    invalid_targets = []
    surface_counts: dict[str, set[tuple[int, int]]] = {}
    for pair in pairs:
        label = str(pair.slice_label)
        counts[label] = counts.get(label, 0) + 1
        if len(tokenizer.encode(pair.y_star, add_special_tokens=False)) != 1:
            invalid_targets.append(pair.y_star)
        distractor = get_prompt_pair_distractor(pair) or ""
        corrupted_prompt = pair.meta.get("pretruncate_corrupted", pair.x_crp)
        count_pair = (
            corrupted_prompt.count(pair.y_star),
            corrupted_prompt.count(distractor) if distractor else 0,
        )
        surface_counts.setdefault(label, set()).add(count_pair)
    if counts != {label: expected_per_class for label in LABELS}:
        raise ValueError(f"Unbalanced suite: {counts}")
    if invalid_targets:
        raise ValueError(f"Multi-token targets: {sorted(set(invalid_targets))}")
    expected_surface = {(2, 1)}
    if any(values != expected_surface for values in surface_counts.values()):
        raise ValueError(f"Surface counts are not balanced: {surface_counts}")
    return {
        "counts": counts,
        "surface_counts": {
            label: sorted(list(values)) for label, values in surface_counts.items()
        },
        "invalid_targets": invalid_targets,
    }


def subset_dataset(tensors: Iterable[PatchEffectTensor]) -> PatchEffectDataset:
    return PatchEffectDataset(tensors=list(tensors))


def make_bags(
    dataset: PatchEffectDataset,
    *,
    split: str,
    bag_size: int,
) -> list[Bag]:
    bags: list[Bag] = []
    for slice_label in sorted(dataset.get_slices(), key=str):
        tensors = dataset.get_by_slice(slice_label)
        if len(tensors) % bag_size:
            raise ValueError(
                f"{slice_label} has {len(tensors)} tensors, not divisible by {bag_size}"
            )
        for bag_index in range(len(tensors) // bag_size):
            start = bag_index * bag_size
            members = tensors[start : start + bag_size]
            bags.append(
                Bag(
                    bag_id=f"{split}|{slice_label}|bag{bag_index:03d}",
                    label=str(slice_label),
                    tensors=members,
                )
            )
    return bags


def assert_paired_bags(first: Sequence[Bag], second: Sequence[Bag]) -> None:
    first_pairs = [(bag.bag_id, bag.label, len(bag.tensors)) for bag in first]
    second_pairs = [(bag.bag_id, bag.label, len(bag.tensors)) for bag in second]
    if first_pairs != second_pairs:
        raise ValueError("Bag IDs, labels, or sizes are not paired across conditions")


def raw_feature_bundles(bags: Sequence[Bag]) -> dict[str, FeatureBundle]:
    means = []
    mean_stds = []
    labels = []
    ids = []
    for bag in bags:
        stack = np.stack(
            [
                np.asarray(tensor.effects, dtype=np.float32).reshape(-1)
                for tensor in bag.tensors
            ]
        )
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        means.append(mean)
        mean_stds.append(np.concatenate([mean, std]))
        labels.append(bag.label)
        ids.append(bag.bag_id)
    mean_matrix = np.stack(means).astype(np.float32)
    mean_std_matrix = np.stack(mean_stds).astype(np.float32)
    return {
        "raw_mean": FeatureBundle(
            mean_matrix,
            labels,
            ids,
            [f"mean_effect_{index}" for index in range(mean_matrix.shape[1])],
        ),
        "raw_mean_std": FeatureBundle(
            mean_std_matrix,
            labels,
            ids,
            [f"mean_std_effect_{index}" for index in range(mean_std_matrix.shape[1])],
        ),
    }


def surface_cue_bundle(bags: Sequence[Bag]) -> FeatureBundle:
    rows = []
    for bag in bags:
        member_rows = []
        for tensor in bag.tensors:
            pair = tensor.prompt_pair
            distractor = get_prompt_pair_distractor(pair) or ""
            corrupted_prompt = pair.meta.get("pretruncate_corrupted", pair.x_crp)
            member_rows.append(
                [
                    float(corrupted_prompt.count(pair.y_star)),
                    float(corrupted_prompt.count(distractor)) if distractor else 0.0,
                    float(len(corrupted_prompt)),
                    float(len(corrupted_prompt.split())),
                    float(len(pair.meta.get("context_clauses", []))),
                ]
            )
        rows.append(np.mean(np.asarray(member_rows), axis=0))
    return FeatureBundle(
        X=np.asarray(rows, dtype=np.float32),
        y=[bag.label for bag in bags],
        ids=[bag.bag_id for bag in bags],
        feature_names=[
            "target_count",
            "distractor_count",
            "character_count",
            "word_count",
            "context_clause_count",
        ],
    )


def build_bag_graphs(
    bags: Sequence[Bag],
    *,
    k: int,
) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
    builder = create_graph_builder("correlation_topk", k=k, enforce_direction=True)
    result = []
    for bag in bags:
        slice_label = bag.tensors[0].prompt_pair.slice_label
        graph = builder.build_from_tensors(
            bag.tensors,
            slice_label,
            graph_role="non_saturated_paired_bag",
            construction_mode="paired_bag_correlation",
            metadata={"bag_id": bag.bag_id, "bag_size": len(bag.tensors)},
        )
        result.append((graph, slice_label))
    return result


def _node_key(node: Any) -> str:
    head = "none" if node.head is None else str(node.head)
    return f"L{node.layer}:T{node.token}:{node.node_type}:H{head}"


def _edge_key(src: Any, dst: Any) -> str:
    return f"edge|{_node_key(src)}->{_node_key(dst)}"


def fixed_edge_bundle(
    graphs: Sequence[tuple[PatchInfluenceGraph, SliceLabel]],
    bags: Sequence[Bag],
    *,
    mode: str,
    feature_names: Sequence[str] | None = None,
) -> FeatureBundle:
    graph_list = [graph for graph, _ in graphs]
    if feature_names is None:
        nodes = graph_list[0].nodes
        feature_names = sorted(
            _edge_key(src, dst) for src in nodes for dst in nodes if src != dst
        )
    names = list(feature_names)
    feature_index = {name: index for index, name in enumerate(names)}
    matrix = np.zeros((len(graph_list), len(names)), dtype=np.float32)
    for row_index, graph in enumerate(graph_list):
        for edge in graph.edges:
            key = _edge_key(graph.nodes[edge.src], graph.nodes[edge.dst])
            column = feature_index.get(key)
            if column is None:
                continue
            if mode == "signed":
                value = float(np.sign(edge.weight))
            elif mode == "weighted":
                value = float(edge.weight)
            else:
                raise ValueError(f"Unknown edge mode: {mode}")
            matrix[row_index, column] += np.float32(value)
    return FeatureBundle(
        matrix,
        [bag.label for bag in bags],
        [bag.bag_id for bag in bags],
        names,
    )


def motif_bundle(
    graphs: Sequence[tuple[PatchInfluenceGraph, SliceLabel]],
    bags: Sequence[Bag],
) -> FeatureBundle:
    matrix = compute_directed_motif_features_from_list(list(graphs))
    return FeatureBundle(
        matrix.to_matrix(),
        [bag.label for bag in bags],
        [bag.bag_id for bag in bags],
        list(matrix.feature_names),
    )


def fit_pca_bundle(
    train: FeatureBundle,
    others: Sequence[FeatureBundle],
    *,
    target_dim: int,
    seed: int,
) -> tuple[FeatureBundle, list[FeatureBundle]]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.X)
    dimension = min(target_dim, train.X.shape[0] - 1, train.X.shape[1])
    if dimension < 1:
        raise ValueError("PCA needs at least two training samples and one feature")
    pca = PCA(n_components=dimension, random_state=seed)
    train_matrix = pca.fit_transform(train_scaled).astype(np.float32)
    feature_names = [f"raw_pca_{index}" for index in range(dimension)]
    transformed_train = FeatureBundle(
        train_matrix,
        list(train.y),
        list(train.ids),
        feature_names,
    )
    transformed_others = [
        FeatureBundle(
            pca.transform(scaler.transform(bundle.X)).astype(np.float32),
            list(bundle.y),
            list(bundle.ids),
            feature_names,
        )
        for bundle in others
    ]
    return transformed_train, transformed_others


def build_all_representations(
    condition_bags: dict[str, list[Bag]],
    *,
    k: int,
    seed: int,
) -> dict[str, dict[str, FeatureBundle]]:
    conditions = ("train", "validation", "test_clean", "test_shift")
    raw_by_condition = {
        condition: raw_feature_bundles(condition_bags[condition])
        for condition in conditions
    }
    graphs_by_condition = {
        condition: build_bag_graphs(condition_bags[condition], k=k)
        for condition in conditions
    }
    representations: dict[str, dict[str, FeatureBundle]] = {
        "raw_mean": {
            condition: raw_by_condition[condition]["raw_mean"]
            for condition in conditions
        },
        "raw_mean_std": {
            condition: raw_by_condition[condition]["raw_mean_std"]
            for condition in conditions
        },
        "surface_cue_control": {
            condition: surface_cue_bundle(condition_bags[condition])
            for condition in conditions
        },
    }

    motif_by_condition = {
        condition: motif_bundle(
            graphs_by_condition[condition],
            condition_bags[condition],
        )
        for condition in conditions
    }
    motif_dim = motif_by_condition["train"].X.shape[1]
    pca_train, pca_others = fit_pca_bundle(
        raw_by_condition["train"]["raw_mean_std"],
        [raw_by_condition[condition]["raw_mean_std"] for condition in conditions[1:]],
        target_dim=motif_dim,
        seed=seed,
    )
    representations["raw_pca_motif"] = {
        "train": pca_train,
        **{
            condition: bundle
            for condition, bundle in zip(conditions[1:], pca_others, strict=True)
        },
    }
    representations["graph_directed_motifs"] = motif_by_condition

    for mode in ("signed", "weighted"):
        name = f"graph_edge_slot_{mode}"
        train_bundle = fixed_edge_bundle(
            graphs_by_condition["train"],
            condition_bags["train"],
            mode=mode,
        )
        representations[name] = {"train": train_bundle}
        for condition in conditions[1:]:
            representations[name][condition] = fixed_edge_bundle(
                graphs_by_condition[condition],
                condition_bags[condition],
                mode=mode,
                feature_names=train_bundle.feature_names,
            )

    train_ids = representations["raw_mean"]["train"].ids
    for representation, bundles in representations.items():
        if bundles["train"].ids != train_ids:
            raise ValueError(f"Training bags differ for {representation}")
        for condition in conditions:
            expected_ids = representations["raw_mean"][condition].ids
            if bundles[condition].ids != expected_ids:
                raise ValueError(
                    f"Paired bag invariant failed for {representation}/{condition}"
                )
    return representations


def symmetric_noise_labels(
    labels: Sequence[str],
    *,
    rate: float,
    seed: int,
) -> list[str]:
    if not 0.0 <= rate < 0.5:
        raise ValueError("Noise rate must be in [0, 0.5)")
    output = list(labels)
    rng = np.random.default_rng(seed)
    classes = sorted(set(labels))
    if len(classes) != 2:
        raise ValueError("Symmetric label noise currently requires two classes")
    for class_index, label in enumerate(classes):
        indices = np.array(
            [index for index, value in enumerate(labels) if value == label]
        )
        flips = int(round(rate * len(indices)))
        if flips:
            selected = rng.choice(indices, size=flips, replace=False)
            replacement = classes[1 - class_index]
            for index in selected:
                output[int(index)] = replacement
    return output


def fit_evaluate(
    train: FeatureBundle,
    evaluation: FeatureBundle,
    *,
    train_labels: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="linear",
                    C=1.0,
                    probability=False,
                    random_state=seed,
                ),
            ),
        ]
    )
    estimator.fit(train.X, list(train_labels))
    predictions = estimator.predict(evaluation.X).tolist()
    return {
        "accuracy": float(accuracy_score(evaluation.y, predictions)),
        "macro_f1": float(f1_score(evaluation.y, predictions, average="macro")),
        "predictions": predictions,
        "y_true": list(evaluation.y),
        "ids": list(evaluation.ids),
    }


def select_representation(
    representations: dict[str, dict[str, FeatureBundle]],
    candidates: Sequence[str],
    *,
    seed: int,
) -> tuple[str, dict[str, dict[str, float]]]:
    scores: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        result = fit_evaluate(
            representations[candidate]["train"],
            representations[candidate]["validation"],
            train_labels=representations[candidate]["train"].y,
            seed=seed,
        )
        scores[candidate] = {
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
        }
    selected = sorted(
        candidates,
        key=lambda name: (
            -scores[name]["macro_f1"],
            -scores[name]["accuracy"],
            name,
        ),
    )[0]
    return selected, scores


def dataset_behavior(dataset: PatchEffectDataset) -> dict[str, float]:
    clean_accuracy = float(np.mean([tensor.clean_score > 0 for tensor in dataset]))
    corrupted_accuracy = float(np.mean([tensor.base_score > 0 for tensor in dataset]))
    return {
        "clean_accuracy": clean_accuracy,
        "corrupted_accuracy": corrupted_accuracy,
        "accuracy_gap": clean_accuracy - corrupted_accuracy,
        "mean_score_gap": float(
            np.mean([tensor.clean_score - tensor.base_score for tensor in dataset])
        ),
        "num_tensors": len(dataset),
    }


def compute_condition_dataset(
    *,
    model: Any,
    pairs: list[PromptPair],
    cache_dir: Path,
    show_progress: bool,
) -> PatchEffectDataset:
    return compute_patch_effects(
        model,
        pairs,
        cache_dir=str(cache_dir),
        show_progress=show_progress,
        node_types=("res",),
    )


def _config_signature(
    args: argparse.Namespace, selected_tier: int | None = None
) -> str:
    payload = {
        "model_name": args.model_name,
        "tail_tokens": args.tail_tokens,
        "bag_size": args.bag_size,
        "train_bags_per_class": args.train_bags_per_class,
        "validation_bags_per_class": args.validation_bags_per_class,
        "test_bags_per_class": args.test_bags_per_class,
        "k": args.k,
        "noise_rates": args.noise_rates,
        "selected_tier": selected_tier,
    }
    return stable_hash(payload)


def load_seed_checkpoint(
    path: Path,
    *,
    config_signature: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_signature") != config_signature:
        raise ValueError(f"Checkpoint configuration mismatch: {path}")
    return payload


def _prepare_split(
    *,
    split: str,
    bags_per_class: int,
    bag_size: int,
    seed: int,
) -> list[PromptPair]:
    return generate_base_pairs(
        split=split,
        n_per_class=bags_per_class * bag_size,
        seed=seed,
    )


def run_pilot(
    args: argparse.Namespace,
    *,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
) -> dict[str, Any]:
    pilot_path = output_dir / "pilot.json"
    pilot_signature = stable_hash(
        {
            "base": _config_signature(args),
            "pilot_seed": args.pilot_seed,
            "pilot_train_bags": args.pilot_train_bags_per_class,
            "pilot_validation_bags": args.pilot_validation_bags_per_class,
        }
    )
    existing = load_seed_checkpoint(
        pilot_path,
        config_signature=pilot_signature,
    )
    if existing is not None and args.resume:
        logging.info("Reusing completed pilot: %s", pilot_path)
        return existing

    train_base = _prepare_split(
        split="train",
        bags_per_class=args.pilot_train_bags_per_class,
        bag_size=args.bag_size,
        seed=args.pilot_seed,
    )
    train_pairs = prepare_condition_pairs(
        train_base,
        split="train",
        tier=-1,
        seed=args.pilot_seed + 1,
        tokenizer=tokenizer,
        tail_tokens=args.tail_tokens,
    )
    validate_pair_suite(
        train_pairs,
        tokenizer,
        expected_per_class=args.pilot_train_bags_per_class * args.bag_size,
    )
    train_dataset = compute_condition_dataset(
        model=model,
        pairs=train_pairs,
        cache_dir=Path(args.cache_root) / "pilot" / "train",
        show_progress=args.show_progress,
    )
    train_bags = make_bags(
        train_dataset,
        split="train",
        bag_size=args.bag_size,
    )
    train_raw = raw_feature_bundles(train_bags)

    validation_base = _prepare_split(
        split="validation",
        bags_per_class=args.pilot_validation_bags_per_class,
        bag_size=args.bag_size,
        seed=args.pilot_seed + 10_000,
    )
    tier_reports = []
    for tier in (1, 2, 3):
        validation_pairs = prepare_condition_pairs(
            validation_base,
            split="validation",
            tier=tier,
            seed=args.pilot_seed + 20_000 + tier,
            tokenizer=tokenizer,
            tail_tokens=args.tail_tokens,
        )
        validate_pair_suite(
            validation_pairs,
            tokenizer,
            expected_per_class=args.pilot_validation_bags_per_class * args.bag_size,
        )
        validation_dataset = compute_condition_dataset(
            model=model,
            pairs=validation_pairs,
            cache_dir=Path(args.cache_root) / "pilot" / f"validation_tier{tier}",
            show_progress=args.show_progress,
        )
        validation_bags = make_bags(
            validation_dataset,
            split="validation",
            bag_size=args.bag_size,
        )
        validation_raw = raw_feature_bundles(validation_bags)
        pca_train, [pca_validation] = fit_pca_bundle(
            train_raw["raw_mean_std"],
            [validation_raw["raw_mean_std"]],
            target_dim=36,
            seed=args.pilot_seed,
        )
        candidates = {
            "raw_mean": (train_raw["raw_mean"], validation_raw["raw_mean"]),
            "raw_mean_std": (
                train_raw["raw_mean_std"],
                validation_raw["raw_mean_std"],
            ),
            "raw_pca_motif": (pca_train, pca_validation),
        }
        scores = {}
        for name, (train_bundle, validation_bundle) in candidates.items():
            evaluated = fit_evaluate(
                train_bundle,
                validation_bundle,
                train_labels=train_bundle.y,
                seed=args.pilot_seed,
            )
            scores[name] = {
                "accuracy": evaluated["accuracy"],
                "macro_f1": evaluated["macro_f1"],
            }
        best_raw = sorted(
            scores,
            key=lambda name: (
                -scores[name]["macro_f1"],
                -scores[name]["accuracy"],
                name,
            ),
        )[0]
        behavior = dataset_behavior(validation_dataset)
        eligible = (
            0.55 <= scores[best_raw]["accuracy"] <= 0.90
            and behavior["accuracy_gap"] >= 0.10
        )
        tier_reports.append(
            {
                "tier": tier,
                "best_raw": best_raw,
                "scores": scores,
                "behavior": behavior,
                "eligible": eligible,
            }
        )
        logging.info(
            "Pilot tier=%d raw_acc=%.3f behavior_gap=%.3f eligible=%s",
            tier,
            scores[best_raw]["accuracy"],
            behavior["accuracy_gap"],
            eligible,
        )

    selected = next((report for report in tier_reports if report["eligible"]), None)
    selected_tier = int(selected["tier"]) if selected else 3
    payload = {
        "config_signature": pilot_signature,
        "pilot_seed": args.pilot_seed,
        "tier_reports": tier_reports,
        "selected_tier": selected_tier,
        "pilot_valid": selected is not None,
    }
    atomic_write_json(pilot_path, payload)
    return payload


def run_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    selected_tier: int,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
) -> dict[str, Any]:
    signature = _config_signature(args, selected_tier)
    checkpoint = output_dir / "seeds" / f"seed_{seed}.json"
    existing = load_seed_checkpoint(checkpoint, config_signature=signature)
    if existing is not None and args.resume:
        logging.info("Reusing completed seed=%d", seed)
        return existing

    split_specs = {
        "train": ("train", args.train_bags_per_class, -1, seed + 1_000),
        "validation": (
            "validation",
            args.validation_bags_per_class,
            selected_tier,
            seed + 2_000,
        ),
        "test_clean": ("test", args.test_bags_per_class, 0, seed + 3_000),
        "test_shift": (
            "test",
            args.test_bags_per_class,
            selected_tier,
            seed + 4_000,
        ),
    }
    test_base = _prepare_split(
        split="test",
        bags_per_class=args.test_bags_per_class,
        bag_size=args.bag_size,
        seed=seed + 30_000,
    )
    base_pairs = {
        "train": _prepare_split(
            split="train",
            bags_per_class=args.train_bags_per_class,
            bag_size=args.bag_size,
            seed=seed + 10_000,
        ),
        "validation": _prepare_split(
            split="validation",
            bags_per_class=args.validation_bags_per_class,
            bag_size=args.bag_size,
            seed=seed + 20_000,
        ),
        "test_clean": test_base,
        "test_shift": test_base,
    }

    datasets: dict[str, PatchEffectDataset] = {}
    condition_bags: dict[str, list[Bag]] = {}
    suite_reports = {}
    behavior_reports = {}
    for condition, (split, bags_per_class, tier, condition_seed) in split_specs.items():
        pairs = prepare_condition_pairs(
            base_pairs[condition],
            split=split,
            tier=tier,
            seed=condition_seed,
            tokenizer=tokenizer,
            tail_tokens=args.tail_tokens,
        )
        suite_reports[condition] = validate_pair_suite(
            pairs,
            tokenizer,
            expected_per_class=bags_per_class * args.bag_size,
        )
        logging.info(
            "seed=%d computing condition=%s tensors=%d", seed, condition, len(pairs)
        )
        dataset = compute_condition_dataset(
            model=model,
            pairs=pairs,
            cache_dir=Path(args.cache_root)
            / "final"
            / args.model_name
            / f"tier{selected_tier}"
            / f"seed{seed}"
            / condition,
            show_progress=args.show_progress,
        )
        datasets[condition] = dataset
        behavior_reports[condition] = dataset_behavior(dataset)
        condition_bags[condition] = make_bags(
            dataset,
            split="test" if condition.startswith("test_") else condition,
            bag_size=args.bag_size,
        )
    assert_paired_bags(condition_bags["test_clean"], condition_bags["test_shift"])

    representations = build_all_representations(
        condition_bags,
        k=args.k,
        seed=seed,
    )
    selected_raw, raw_validation = select_representation(
        representations,
        RAW_REPRESENTATIONS,
        seed=seed,
    )
    selected_graph, graph_validation = select_representation(
        representations,
        GRAPH_REPRESENTATIONS,
        seed=seed,
    )
    logging.info(
        "seed=%d selected_raw=%s selected_graph=%s",
        seed,
        selected_raw,
        selected_graph,
    )

    rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    noise_rates = parse_floats(args.noise_rates)
    for representation in ALL_REPRESENTATIONS:
        train_bundle = representations[representation]["train"]
        predictions[representation] = {}
        for noise_rate in noise_rates:
            noise_key = f"{noise_rate:.1f}"
            noisy_labels = symmetric_noise_labels(
                train_bundle.y,
                rate=noise_rate,
                seed=seed + int(round(noise_rate * 10_000)),
            )
            predictions[representation][noise_key] = {}
            for condition in ("test_clean", "test_shift"):
                evaluated = fit_evaluate(
                    train_bundle,
                    representations[representation][condition],
                    train_labels=noisy_labels,
                    seed=seed,
                )
                rows.append(
                    {
                        "seed": seed,
                        "representation": representation,
                        "selected_variant": "",
                        "noise_rate": noise_rate,
                        "condition": condition,
                        "accuracy": evaluated["accuracy"],
                        "macro_f1": evaluated["macro_f1"],
                        "num_features": train_bundle.X.shape[1],
                        "num_train": train_bundle.X.shape[0],
                        "num_test": representations[representation][condition].X.shape[
                            0
                        ],
                    }
                )
                predictions[representation][noise_key][condition] = evaluated

    for selected_name, selected_variant in (
        ("selected_raw", selected_raw),
        ("selected_graph", selected_graph),
    ):
        predictions[selected_name] = predictions[selected_variant]
        for row in list(rows):
            if row["representation"] == selected_variant:
                selected_row = dict(row)
                selected_row["representation"] = selected_name
                selected_row["selected_variant"] = selected_variant
                rows.append(selected_row)

    payload = {
        "config_signature": signature,
        "seed": seed,
        "selected_tier": selected_tier,
        "selected_raw": selected_raw,
        "selected_graph": selected_graph,
        "raw_validation": raw_validation,
        "graph_validation": graph_validation,
        "suite_reports": suite_reports,
        "behavior_reports": behavior_reports,
        "rows": rows,
        "predictions": predictions,
    }
    atomic_write_json(checkpoint, payload)
    return payload


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["representation"]),
            float(row["noise_rate"]),
            str(row["condition"]),
        )
        groups.setdefault(key, []).append(row)
    summary = []
    for (representation, noise_rate, condition), values in sorted(groups.items()):
        accuracy = np.array([float(row["accuracy"]) for row in values])
        macro_f1 = np.array([float(row["macro_f1"]) for row in values])
        features = np.array([int(row["num_features"]) for row in values])
        summary.append(
            {
                "representation": representation,
                "noise_rate": noise_rate,
                "condition": condition,
                "accuracy_mean": float(accuracy.mean()),
                "accuracy_std": float(accuracy.std()),
                "macro_f1_mean": float(macro_f1.mean()),
                "macro_f1_std": float(macro_f1.std()),
                "num_features_mean": float(features.mean()),
                "seeds": len(values),
            }
        )
    return summary


def _metric_from_predictions(
    y_true: Sequence[str], predictions: Sequence[str], metric: str
) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, predictions))
    if metric == "macro_f1":
        return float(f1_score(y_true, predictions, average="macro"))
    raise ValueError(f"Unknown metric: {metric}")


def paired_bootstrap_delta(
    seed_payloads: Sequence[dict[str, Any]],
    *,
    metric: str,
    noise_rate: float,
    condition: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    noise_key = f"{noise_rate:.1f}"
    point_deltas = []
    for payload in seed_payloads:
        raw = payload["predictions"]["selected_raw"][noise_key][condition]
        graph = payload["predictions"]["selected_graph"][noise_key][condition]
        point_deltas.append(
            _metric_from_predictions(graph["y_true"], graph["predictions"], metric)
            - _metric_from_predictions(raw["y_true"], raw["predictions"], metric)
        )

    bootstrap_values = np.zeros(samples, dtype=np.float64)
    for sample_index in range(samples):
        sampled_seed_indices = rng.integers(
            0, len(seed_payloads), size=len(seed_payloads)
        )
        sampled_seed_deltas: list[float] = []
        for payload_index in sampled_seed_indices:
            payload = seed_payloads[int(payload_index)]
            raw = payload["predictions"]["selected_raw"][noise_key][condition]
            graph = payload["predictions"]["selected_graph"][noise_key][condition]
            labels = np.asarray(raw["y_true"])
            seed_true: list[str] = []
            seed_raw_predictions: list[str] = []
            seed_graph_predictions: list[str] = []
            for label in sorted(set(labels.tolist())):
                indices = np.flatnonzero(labels == label)
                sampled = rng.choice(indices, size=len(indices), replace=True)
                seed_true.extend(labels[sampled].tolist())
                seed_raw_predictions.extend(
                    np.asarray(raw["predictions"])[sampled].tolist()
                )
                seed_graph_predictions.extend(
                    np.asarray(graph["predictions"])[sampled].tolist()
                )
            sampled_seed_deltas.append(
                _metric_from_predictions(seed_true, seed_graph_predictions, metric)
                - _metric_from_predictions(seed_true, seed_raw_predictions, metric)
            )
        bootstrap_values[sample_index] = float(np.mean(sampled_seed_deltas))
    return {
        "mean_seed_delta": float(np.mean(point_deltas)),
        "ci_low": float(np.quantile(bootstrap_values, 0.025)),
        "ci_high": float(np.quantile(bootstrap_values, 0.975)),
    }


def evaluate_gate(
    seed_payloads: Sequence[dict[str, Any]],
    *,
    pilot_valid: bool,
    bootstrap_samples: int,
) -> dict[str, Any]:
    shift_accuracy = paired_bootstrap_delta(
        seed_payloads,
        metric="accuracy",
        noise_rate=0.0,
        condition="test_shift",
        samples=bootstrap_samples,
        seed=9101,
    )
    shift_f1 = paired_bootstrap_delta(
        seed_payloads,
        metric="macro_f1",
        noise_rate=0.0,
        condition="test_shift",
        samples=bootstrap_samples,
        seed=9102,
    )
    noise_f1 = paired_bootstrap_delta(
        seed_payloads,
        metric="macro_f1",
        noise_rate=0.2,
        condition="test_shift",
        samples=bootstrap_samples,
        seed=9103,
    )
    wins = 0
    clean_graph_accuracies = []
    for payload in seed_payloads:
        raw_shift = payload["predictions"]["selected_raw"]["0.0"]["test_shift"]
        graph_shift = payload["predictions"]["selected_graph"]["0.0"]["test_shift"]
        raw_accuracy = _metric_from_predictions(
            raw_shift["y_true"], raw_shift["predictions"], "accuracy"
        )
        graph_accuracy = _metric_from_predictions(
            graph_shift["y_true"], graph_shift["predictions"], "accuracy"
        )
        raw_f1 = _metric_from_predictions(
            raw_shift["y_true"], raw_shift["predictions"], "macro_f1"
        )
        graph_f1 = _metric_from_predictions(
            graph_shift["y_true"], graph_shift["predictions"], "macro_f1"
        )
        wins += int(graph_accuracy > raw_accuracy and graph_f1 > raw_f1)
        graph_clean = payload["predictions"]["selected_graph"]["0.0"]["test_clean"]
        clean_graph_accuracies.append(
            _metric_from_predictions(
                graph_clean["y_true"], graph_clean["predictions"], "accuracy"
            )
        )
    clean_graph_accuracy = float(np.mean(clean_graph_accuracies))
    checks = {
        "pilot_valid": bool(pilot_valid),
        "shift_accuracy_delta_at_least_0_05": shift_accuracy["mean_seed_delta"] >= 0.05,
        "shift_accuracy_ci_above_zero": shift_accuracy["ci_low"] > 0.0,
        "shift_f1_delta_at_least_0_05": shift_f1["mean_seed_delta"] >= 0.05,
        "shift_f1_ci_above_zero": shift_f1["ci_low"] > 0.0,
        "noise_20_f1_delta_at_least_0_03": noise_f1["mean_seed_delta"] >= 0.03,
        "wins_at_least_four_of_five": wins >= 4,
        "clean_accuracy_below_0_95": clean_graph_accuracy < 0.95,
    }
    return {
        "decision": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "shift_accuracy_delta": shift_accuracy,
        "shift_macro_f1_delta": shift_f1,
        "noise_20_shift_macro_f1_delta": noise_f1,
        "seed_wins": wins,
        "selected_graph_clean_accuracy_mean": clean_graph_accuracy,
    }


def _summary_lookup(
    summary: Sequence[dict[str, Any]],
    representation: str,
    noise_rate: float,
    condition: str,
) -> dict[str, Any]:
    for row in summary:
        if (
            row["representation"] == representation
            and math.isclose(float(row["noise_rate"]), noise_rate)
            and row["condition"] == condition
        ):
            return row
    raise KeyError((representation, noise_rate, condition))


def _mean_std(row: dict[str, Any], prefix: str) -> str:
    return f"{row[prefix + '_mean']:.3f} ± {row[prefix + '_std']:.3f}"


def build_rebuttal_markdown(
    summary: Sequence[dict[str, Any]],
    seed_payloads: Sequence[dict[str, Any]],
    gate: dict[str, Any],
    *,
    selected_tier: int,
    bag_size: int,
    model_name: str = "gpt2",
    k: int = 5,
) -> str:
    display_names = {
        "raw_mean": "Raw tensor mean",
        "raw_mean_std": "Raw tensor mean+std",
        "raw_pca_motif": "Raw tensor PCA (motif-matched)",
        "graph_edge_slot_signed": "Graph edge-slot (signed)",
        "graph_edge_slot_weighted": "Graph edge-slot (weighted)",
        "graph_directed_motifs": "Graph directed motifs",
        "surface_cue_control": "Prompt surface-cue control",
        "selected_raw": "Selected raw (validation)",
        "selected_graph": "Selected graph (validation)",
    }
    order = [
        "raw_mean",
        "raw_mean_std",
        "raw_pca_motif",
        "graph_edge_slot_signed",
        "graph_edge_slot_weighted",
        "graph_directed_motifs",
        "surface_cue_control",
        "selected_raw",
        "selected_graph",
    ]
    lines = [
        "# Non-saturated raw patch-effect vs graph comparison",
        "",
        f"**Preregistered decision: {gate['decision']}**",
        "",
        "| Representation | Features | Clean Acc. | Clean macro-F1 | Context-shift Acc. | Context-shift macro-F1 | 20% noise + shift Acc. | 20% noise + shift macro-F1 | Δ shifted F1 vs selected raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    selected_raw_by_seed = {}
    for payload in seed_payloads:
        selected_raw_by_seed[int(payload["seed"])] = {
            row["condition"]: row
            for row in payload["rows"]
            if row["representation"] == "selected_raw"
            and math.isclose(float(row["noise_rate"]), 0.0)
        }
    for representation in order:
        clean = _summary_lookup(summary, representation, 0.0, "test_clean")
        shift = _summary_lookup(summary, representation, 0.0, "test_shift")
        noise = _summary_lookup(summary, representation, 0.2, "test_shift")
        deltas = []
        for payload in seed_payloads:
            seed = int(payload["seed"])
            current = next(
                row
                for row in payload["rows"]
                if row["representation"] == representation
                and math.isclose(float(row["noise_rate"]), 0.0)
                and row["condition"] == "test_shift"
            )
            deltas.append(
                float(current["macro_f1"])
                - float(selected_raw_by_seed[seed]["test_shift"]["macro_f1"])
            )
        label = display_names[representation]
        if representation == "selected_graph":
            label = f"**{label}**"
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{clean['num_features_mean']:.0f}",
                    _mean_std(clean, "accuracy"),
                    _mean_std(clean, "macro_f1"),
                    _mean_std(shift, "accuracy"),
                    _mean_std(shift, "macro_f1"),
                    _mean_std(noise, "accuracy"),
                    _mean_std(noise, "macro_f1"),
                    f"{np.mean(deltas):+.3f} ± {np.std(deltas):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "Results are mean ± population standard deviation across "
                f"{len(seed_payloads)} {model_name} seed(s), with {bag_size} disjoint "
                "patch-effect tensors per classifier sample, residual-stream nodes, "
                f"correlation-top-k (k={k}), and a "
                "standardized linear SVM (C=1). "
                + (
                    f"Context tier {selected_tier} was chosen on pilot "
                    "raw-validation performance only. "
                    if gate["checks"].get("pilot_valid", True)
                    else (
                        f"No context tier passed the pilot criteria; tier "
                        f"{selected_tier} is the preregistered fallback. "
                    )
                )
                + "Label noise is applied symmetrically to training labels; all test "
                "labels remain clean."
            ),
            "",
            (
                "Selected-graph minus selected-raw context-shift deltas: "
                f"accuracy {gate['shift_accuracy_delta']['mean_seed_delta']:+.3f} "
                f"[95% CI {gate['shift_accuracy_delta']['ci_low']:+.3f}, "
                f"{gate['shift_accuracy_delta']['ci_high']:+.3f}]; macro-F1 "
                f"{gate['shift_macro_f1_delta']['mean_seed_delta']:+.3f} "
                f"[95% CI {gate['shift_macro_f1_delta']['ci_low']:+.3f}, "
                f"{gate['shift_macro_f1_delta']['ci_high']:+.3f}]."
            ),
            "",
            "Gate checks: " + json.dumps(gate["checks"], sort_keys=True),
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", default=",".join(map(str, FINAL_SEEDS)))
    parser.add_argument("--pilot-seed", type=int, default=2026)
    parser.add_argument("--tail-tokens", type=int, default=16)
    parser.add_argument("--bag-size", type=int, default=12)
    parser.add_argument("--train-bags-per-class", type=int, default=24)
    parser.add_argument("--validation-bags-per-class", type=int, default=8)
    parser.add_argument("--test-bags-per-class", type=int, default=12)
    parser.add_argument("--pilot-train-bags-per-class", type=int, default=8)
    parser.add_argument("--pilot-validation-bags-per-class", type=int, default=4)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--noise-rates", default="0.0,0.1,0.2,0.3")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--cache-root",
        default=".cache/non_saturated_raw_vs_graph",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--fixed-tier",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="Skip pilot tier selection and use this tier; intended for smoke tests.",
    )
    parser.add_argument("--pilot-only", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    noise_rates = parse_floats(args.noise_rates)
    if tuple(noise_rates) != NOISE_RATES:
        logging.warning("Using non-default noise rates: %s", noise_rates)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs")
        / "non_saturated_raw_vs_graph"
        / f"{args.model_name}_{utc_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log", mode="a", encoding="utf-8"),
        ],
    )
    config = {
        **vars(args),
        "output_dir": str(output_dir),
        "seeds_resolved": parse_ints(args.seeds),
        "name_splits": {key: list(value) for key, value in NAME_SPLITS.items()},
        "template_indices": {
            key: list(value) for key, value in TEMPLATE_INDICES.items()
        },
        "representations": list(ALL_REPRESENTATIONS),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip(),
    }
    atomic_write_json(output_dir / "config.json", config)

    logging.info("Loading model=%s device=%s", args.model_name, args.device)
    model = create_model(args.model_name, device=args.device)
    tokenizer = getattr(model, "tokenizer", None) or ToyTokenizerAdapter(model)
    if args.fixed_tier is None:
        pilot = run_pilot(
            args,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
        )
    else:
        pilot_path = output_dir / "pilot.json"
        existing_pilot = (
            json.loads(pilot_path.read_text(encoding="utf-8"))
            if args.resume and pilot_path.exists()
            else None
        )
        if (
            existing_pilot is not None
            and int(existing_pilot.get("selected_tier", -1)) == args.fixed_tier
        ):
            pilot = existing_pilot
        else:
            pilot = {
                "selected_tier": args.fixed_tier,
                "pilot_valid": False,
                "fixed_tier_override": True,
            }
            atomic_write_json(pilot_path, pilot)
    if args.pilot_only:
        logging.info("Pilot complete: %s", output_dir / "pilot.json")
        return

    selected_tier = int(pilot["selected_tier"])
    seed_payloads = []
    for seed in parse_ints(args.seeds):
        seed_payloads.append(
            run_seed(
                args,
                seed=seed,
                selected_tier=selected_tier,
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
            )
        )
    all_rows = [row for payload in seed_payloads for row in payload["rows"]]
    write_csv(output_dir / "detailed_results.csv", all_rows)
    summary = summarize_rows(all_rows)
    write_csv(output_dir / "summary.csv", summary)
    gate = evaluate_gate(
        seed_payloads,
        pilot_valid=bool(pilot.get("pilot_valid")),
        bootstrap_samples=args.bootstrap_samples,
    )
    aggregate = {
        "selected_tier": selected_tier,
        "pilot": pilot,
        "summary": summary,
        "gate": gate,
        "completed_seeds": [int(payload["seed"]) for payload in seed_payloads],
    }
    atomic_write_json(output_dir / "aggregate.json", aggregate)
    markdown = build_rebuttal_markdown(
        summary,
        seed_payloads,
        gate,
        selected_tier=selected_tier,
        bag_size=args.bag_size,
        model_name=args.model_name,
        k=args.k,
    )
    atomic_write_text(output_dir / "rebuttal_table.md", markdown)
    logging.info("Decision=%s", gate["decision"])
    logging.info("Wrote %s", output_dir / "rebuttal_table.md")


if __name__ == "__main__":
    main()
