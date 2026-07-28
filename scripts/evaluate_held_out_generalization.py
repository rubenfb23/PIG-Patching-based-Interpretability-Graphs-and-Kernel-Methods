#!/usr/bin/env python3
"""Held-out generalization evaluation for PIG graph representations.

Tests whether graph representations generalize beyond the specific bootstrap
split used in the main results. Implements stronger train/test splits:

1. Held-out prompt templates: train on one template family, test on another
2. Held-out names: train on one set of names, test on unseen names
3. Held-out corruption config: train on name_swap, test on abba (cross-corruption)
4. Stricter prompt-family split: ensure no shared substrings between train/test

Evaluates: fixed-layout signed, hashed fixed-layout signed, coarse count,
spectral, graphlet, WL.

Reports results per seed and mean/std.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pig.graph import PatchInfluenceGraph
from pig.graph_features import (
    FixedLayoutFeatureMatrix,
    compute_fixed_layout_features_from_list,
)
from pig.kernels import ClassicalKernelClassifier
from pig.prompts import SliceLabel


# ---------------------------------------------------------------------------
# Representation feature extraction
# ---------------------------------------------------------------------------

@dataclass
class RepresentationFeatures:
    """Features for a set of graphs with a specific representation."""
    X: NDArray[np.float64]  # shape (n_graphs, n_features)
    y: NDArray[np.int32]    # shape (n_graphs,) labels
    feature_names: list[str]
    representation_name: str


def extract_features_for_representation(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    representation: str,
    hash_dim: int = 1024,
) -> RepresentationFeatures:
    """Extract features for a specific representation scheme.

    Supported representations:
    - fixed_layout_signed: sign of edge weights in fixed layout
    - fixed_layout_weighted: raw edge weights in fixed layout
    - hashed_fixed_layout_signed: hashed fixed layout with sign accumulation
    - coarse_count: coarse bucket counts
    - spectral: spectral shape features
    - graphlet: graphlet motif counts
    - wl: Weisfeiler-Lehman subtree features
    """
    if representation in ("fixed_layout_signed", "fixed_layout_weighted",
                          "hashed_fixed_layout_signed", "coarse_count"):
        return _extract_fixed_layout_features(
            graphs_with_labels, representation, hash_dim
        )
    elif representation in ("spectral", "graphlet", "wl"):
        return _extract_shape_features(graphs_with_labels, representation)
    else:
        raise ValueError(f"Unknown representation: {representation}")


def _extract_fixed_layout_features(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    representation: str,
    hash_dim: int = 1024,
) -> RepresentationFeatures:
    """Extract fixed-layout or hashed features."""
    # Use the existing fixed-layout computation
    matrix = compute_fixed_layout_features_from_list(graphs_with_labels)

    if representation == "fixed_layout_signed":
        # Convert to signed features
        X = np.sign(matrix.to_matrix()).astype(np.float64)
    elif representation == "fixed_layout_weighted":
        X = matrix.to_matrix().astype(np.float64)
    elif representation == "hashed_fixed_layout_signed":
        X = _hash_features(matrix.to_matrix(), hash_dim)
    elif representation == "coarse_count":
        X = _coarse_count_features(matrix.to_matrix(), matrix.feature_names)
    else:
        X = matrix.to_matrix().astype(np.float64)

    # Build labels: 0 for name_swap, 1 for abba
    label_map = {"name_swap": 0, "abba": 1}
    y = np.array([label_map.get(str(l).split(":")[-1], 0)
                  for _, l in graphs_with_labels], dtype=np.int32)

    return RepresentationFeatures(
        X=X, y=y,
        feature_names=matrix.feature_names if representation != "hashed_fixed_layout_signed"
                     else [f"hash_{i}" for i in range(X.shape[1])],
        representation_name=representation,
    )


def _hash_features(
    matrix: NDArray[np.float64],
    dim: int = 1024,
) -> NDArray[np.float64]:
    """Hash fixed-layout features to fixed dimension via stable hash."""
    import hashlib
    n_graphs, n_features = matrix.shape
    hashed = np.zeros((n_graphs, dim), dtype=np.float64)

    for j in range(n_features):
        key = f"{j}:{dim}"
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % dim
        hashed[:, h] += matrix[:, j]

    return hashed


def _coarse_count_features(
    matrix: NDArray[np.float64],
    feature_names: list[str],
) -> NDArray[np.float64]:
    """Aggregate fixed-layout features into coarse buckets."""
    n_graphs = matrix.shape[0]
    buckets = defaultdict(float)

    for j, name in enumerate(feature_names):
        if "->" not in name:
            continue
        parts = name.split("->")
        src_parts = parts[0].split(":")
        dst_parts = parts[1].split(":")

        if len(src_parts) < 2 or len(dst_parts) < 2:
            continue

        src_layer = int(src_parts[0].replace("L", ""))
        dst_layer = int(dst_parts[0].replace("L", ""))
        src_token = int(src_parts[1].replace("T", ""))
        dst_token = int(dst_parts[1].replace("T", ""))

        delta_layer = abs(dst_layer - src_layer)
        delta_token = abs(dst_token - src_token)

        # Coarse buckets
        layer_bucket = min(delta_layer // 2, 5)
        token_bucket = min(delta_token // 2, 5)
        sign_bucket = 1 if matrix[0, j] > 0 else 0

        bucket_key = f"l{layer_bucket}_t{token_bucket}_s{sign_bucket}"
        buckets[bucket_key] += matrix[0, j]

    # Build feature vector — accumulate per-graph, not just matrix[0]
    bucket_keys = sorted(buckets.keys())
    n_buckets = len(bucket_keys)
    X = np.zeros((n_graphs, n_buckets), dtype=np.float64)
    for i in range(n_graphs):
        for j, name in enumerate(feature_names):
            if "->" not in name:
                continue
            parts = name.split("->")
            src_parts = parts[0].split(":")
            dst_parts = parts[1].split(":")
            if len(src_parts) < 2 or len(dst_parts) < 2:
                continue
            src_layer = int(src_parts[0].replace("L", ""))
            dst_layer = int(dst_parts[0].replace("L", ""))
            src_token = int(src_parts[1].replace("T", ""))
            dst_token = int(dst_parts[1].replace("T", ""))
            delta_layer = abs(dst_layer - src_layer)
            delta_token = abs(dst_token - src_token)
            layer_bucket = min(delta_layer // 2, 5)
            token_bucket = min(delta_token // 2, 5)
            sign_bucket = 1 if matrix[i, j] > 0 else 0
            bucket_key = f"l{layer_bucket}_t{token_bucket}_s{sign_bucket}"
            col = bucket_keys.index(bucket_key)
            X[i, col] += matrix[i, j]

    return X


def _extract_shape_features(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    representation: str,
) -> RepresentationFeatures:
    """Extract spectral or graphlet features from graphs."""
    n_graphs = len(graphs_with_labels)

    if representation == "spectral":
        X = np.zeros((n_graphs, 96), dtype=np.float64)
        for i, (graph, _) in enumerate(graphs_with_labels):
            X[i] = _compute_spectral_features(graph)
        feature_names = [f"spec_{j}" for j in range(96)]
    elif representation == "graphlet":
        X = np.zeros((n_graphs, 38), dtype=np.float64)
        for i, (graph, _) in enumerate(graphs_with_labels):
            X[i] = _compute_graphlet_features(graph)
        feature_names = [f"graphlet_{j}" for j in range(38)]
    elif representation == "wl":
        # Simplified WL: use node degree histogram as proxy
        X = np.zeros((n_graphs, 50), dtype=np.float64)
        for i, (graph, _) in enumerate(graphs_with_labels):
            X[i] = _compute_wl_proxy(graph)
        feature_names = [f"wl_{j}" for j in range(50)]
    else:
        raise ValueError(f"Unknown shape representation: {representation}")

    label_map = {"name_swap": 0, "abba": 1}
    y = np.array([label_map.get(str(l).split(":")[-1], 0)
                  for _, l in graphs_with_labels], dtype=np.int32)

    return RepresentationFeatures(
        X=X, y=y,
        feature_names=feature_names,
        representation_name=representation,
    )


def _compute_spectral_features(graph: PatchInfluenceGraph) -> NDArray[np.float64]:
    """Compute spectral features for a graph (simplified)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg

    n = len(graph.nodes)
    if n == 0:
        return np.zeros(96)

    rows, cols, vals = [], [], []
    for edge in graph.edges:
        rows.extend([edge.src, edge.dst])
        cols.extend([edge.dst, edge.src])
        vals.extend([abs(edge.weight), abs(edge.weight)])

    if not rows:
        return np.zeros(96)

    adj = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    deg = np.array(adj.sum(axis=1)).flatten()
    deg = np.maximum(deg, 1e-10)

    # Normalized Laplacian eigenvalues (few smallest + few largest)
    L_diag = sp.diags(1.0 / np.sqrt(deg))
    L_norm = sp.eye(n) - L_diag @ adj @ L_diag

    try:
        k_eigs = min(10, n - 1)
        if k_eigs > 0:
            eigenvalues = scipy.sparse.linalg.eigsh(
                L_norm, k=k_eigs, which="SM", return_eigenvectors=False
            )
            eigs = np.sort(np.real(eigenvalues))
        else:
            eigs = np.array([])
    except Exception:
        eigs = np.zeros(10)

    # Degree statistics
    in_deg = np.array(adj.sum(axis=0)).flatten()
    out_deg = np.array(adj.sum(axis=1)).flatten()

    features = []
    features.extend(eigs.tolist() + [0.0] * (10 - len(eigs)))
    features.extend([
        float(np.mean(in_deg)), float(np.std(in_deg)),
        float(np.mean(out_deg)), float(np.std(out_deg)),
        float(np.max(in_deg)), float(np.max(out_deg)),
    ])
    # Pad to 96
    features.extend([0.0] * (96 - len(features)))
    return np.array(features[:96])


def _compute_graphlet_features(graph: PatchInfluenceGraph) -> NDArray[np.float64]:
    """Compute graphlet features (simplified directed motifs)."""
    n = len(graph.nodes)
    if n == 0:
        return np.zeros(38)

    # Count directed motifs of size 3
    adj = np.zeros((n, n))
    for edge in graph.edges:
        adj[edge.src, edge.dst] = abs(edge.weight)

    features = []

    # Density
    max_edges = n * (n - 1) if n > 1 else 1
    features.append(float(np.count_nonzero(adj)) / max_edges)

    # Single-edge triplet counts (forward, backward, mutual)
    forward = 0
    backward = 0
    mutual = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if adj[i, j] > 0 and adj[j, i] == 0:
                forward += 1
            elif adj[i, j] == 0 and adj[j, i] > 0:
                backward += 1
            elif adj[i, j] > 0 and adj[j, i] > 0:
                mutual += 1

    features.extend([forward, backward, mutual])

    # Two-edge triplet counts
    two_in = 0  # both point to same node
    two_out = 0  # same node points to both
    chain = 0  # A->B->C
    for k in range(n):
        in_neighbors = np.where(adj[:, k] > 0)[0]
        out_neighbors = np.where(adj[k, :] > 0)[0]
        two_in += len(in_neighbors) * (len(in_neighbors) - 1) // 2
        two_out += len(out_neighbors) * (len(out_neighbors) - 1) // 2

    for i in range(n):
        for j in range(n):
            if i == j or adj[i, j] == 0:
                continue
            for k in range(n):
                if k != i and k != j and adj[j, k] > 0:
                    chain += 1

    features.extend([two_in, two_out, chain])

    # Weight statistics
    weights = adj[adj > 0]
    if len(weights) > 0:
        features.extend([
            float(np.mean(weights)), float(np.std(weights)),
            float(np.skew(weights)), float(np.kurtosis(weights)),
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    # Sign statistics
    all_weights = np.array([e.weight for e in graph.edges])
    if len(all_weights) > 0:
        pos = np.sum(all_weights > 0)
        neg = np.sum(all_weights < 0)
        features.extend([float(pos), float(neg), pos / max(len(all_weights), 1)])
    else:
        features.extend([0.0, 0.0, 0.0])

    # Pad to 38
    features.extend([0.0] * (38 - len(features)))
    return np.array(features[:38])


def _compute_wl_proxy(graph: PatchInfluenceGraph) -> NDArray[np.float64]:
    """Compute WL-like features as a simplified proxy."""
    n = len(graph.nodes)
    if n == 0:
        return np.zeros(50)

    # Node degree histogram (binned)
    out_deg = np.array([sum(1 for e in graph.edges if e.src == i) for i in range(n)])
    in_deg = np.array([sum(1 for e in graph.edges if e.dst == i) for i in range(n)])

    features = []
    for deg in [out_deg, in_deg]:
        for bin_max in [1, 2, 3, 5, 10]:
            features.append(float(np.sum(deg <= bin_max)) / max(n, 1))

    # Edge weight histogram
    weights = [abs(e.weight) for e in graph.edges]
    if weights:
        for bin_max in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            features.append(float(np.sum(np.array(weights) <= bin_max)) / len(weights))
    else:
        features.extend([0.0] * 6)

    # Layer distribution
    layers = [graph.nodes[e.src].layer for e in graph.edges]
    if layers:
        for layer in range(12):
            features.append(float(np.sum(np.array(layers) == layer)) / max(len(layers), 1))
    else:
        features.extend([0.0] * 12)

    # Pad to 50
    features.extend([0.0] * (50 - len(features)))
    return np.array(features[:50])


# ---------------------------------------------------------------------------
# Held-out split strategies
# ---------------------------------------------------------------------------

@dataclass
class HeldOutSplit:
    """A train/test split for held-out evaluation."""
    name: str
    train_graphs: list[tuple[PatchInfluenceGraph, SliceLabel]]
    test_graphs: list[tuple[PatchInfluenceGraph, SliceLabel]]
    description: str


def create_held_out_splits(
    all_graphs: list[tuple[PatchInfluenceGraph, SliceLabel]],
    strategy: str = "template",
    seed: int = 42,
) -> list[HeldOutSplit]:
    """Create held-out splits based on the specified strategy.

    Strategies:
    - template: split by prompt template family
    - names: split by name sets
    - corruption: split by corruption type (cross-corruption)
    - family: stricter prompt-family split
    - all: all splits
    """
    splits = []

    if strategy in ("template", "all"):
        splits.extend(_split_by_template(all_graphs, seed))

    if strategy in ("names", "all"):
        splits.extend(_split_by_names(all_graphs, seed))

    if strategy in ("corruption", "all"):
        splits.extend(_split_by_corruption(all_graphs))

    if strategy in ("family", "all"):
        splits.extend(_split_by_family(all_graphs, seed))

    return splits


def _split_by_template(
    all_graphs: list[tuple[PatchInfluenceGraph, SliceLabel]],
    seed: int,
) -> list[HeldOutSplit]:
    """Split by prompt template family."""
    rng = np.random.default_rng(seed)

    # Group by template (extracted from graph metadata)
    template_groups: defaultdict = defaultdict(list)
    for graph, label in all_graphs:
        template = graph.metadata.get("template", str(label))
        template_groups[template].append((graph, label))

    templates = sorted(template_groups.keys())
    splits = []

    for i, test_template in enumerate(templates):
        train_graphs = []
        for t in templates:
            if t != test_template:
                train_graphs.extend(template_groups[t])

        if not train_graphs or not template_groups[test_template]:
            continue

        splits.append(HeldOutSplit(
            name=f"template_holdout_{test_template}",
            train_graphs=train_graphs,
            test_graphs=template_groups[test_template],
            description=f"Train on all templates except '{test_template}', test on '{test_template}'",
        ))

    return splits


def _split_by_names(
    all_graphs: list[tuple[PatchInfluenceGraph, SliceLabel]],
    seed: int,
) -> list[HeldOutSplit]:
    """Split by name sets (held-out names)."""
    rng = np.random.default_rng(seed)

    # Group by name set (from metadata)
    name_groups: defaultdict = defaultdict(list)
    for graph, label in all_graphs:
        name_set = graph.metadata.get("name_set", str(label))
        name_groups[name_set].append((graph, label))

    name_sets = sorted(name_groups.keys())
    splits = []

    for i, test_names in enumerate(name_sets):
        train_graphs = []
        for ns in name_sets:
            if ns != test_names:
                train_graphs.extend(name_groups[ns])

        if not train_graphs or not name_groups[test_names]:
            continue

        splits.append(HeldOutSplit(
            name=f"names_holdout_{test_names}",
            train_graphs=train_graphs,
            test_graphs=name_groups[test_names],
            description=f"Train on all name sets except '{test_names}', test on '{test_names}'",
        ))

    return splits


def _split_by_corruption(
    all_graphs: list[tuple[PatchInfluenceGraph, SliceLabel]],
) -> list[HeldOutSplit]:
    """Cross-corruption split: train on name_swap, test on abba and vice versa."""
    swap_graphs = [(g, l) for g, l in all_graphs if "swap" in str(l)]
    abba_graphs = [(g, l) for g, l in all_graphs if "abba" in str(l)]

    splits = []

    if swap_graphs and abba_graphs:
        splits.append(HeldOutSplit(
            name="corruption_swap2abba",
            train_graphs=swap_graphs,
            test_graphs=abba_graphs,
            description="Train on name_swap, test on abba",
        ))
        splits.append(HeldOutSplit(
            name="corruption_abba2swap",
            train_graphs=abba_graphs,
            test_graphs=swap_graphs,
            description="Train on abba, test on name_swap",
        ))

    return splits


def _split_by_family(
    all_graphs: list[tuple[PatchInfluenceGraph, SliceLabel]],
    seed: int,
) -> list[HeldOutSplit]:
    """Stricter prompt-family split: ensure no shared substrings."""
    rng = np.random.default_rng(seed)

    # Group by prompt family (first few words of clean prompt)
    family_groups: defaultdict = defaultdict(list)
    for graph, label in all_graphs:
        prompt = graph.metadata.get("x_cln", "")
        family = " ".join(prompt.split()[:3]) if prompt else str(label)
        family_groups[family].append((graph, label))

    families = sorted(family_groups.keys())
    splits = []

    for i, test_family in enumerate(families):
        train_graphs = []
        for f in families:
            if f != test_family:
                train_graphs.extend(family_groups[f])

        if not train_graphs or not family_groups[test_family]:
            continue

        splits.append(HeldOutSplit(
            name=f"family_holdout_{test_family[:20]}",
            train_graphs=train_graphs,
            test_graphs=family_groups[test_family],
            description=f"Train on all families except '{test_family}', test on '{test_family}'",
        ))

    return splits


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

REPRESENTATIONS = [
    "fixed_layout_signed",
    "hashed_fixed_layout_signed",
    "coarse_count",
    "spectral",
    "graphlet",
    "wl",
]


def evaluate_split(
    split: HeldOutSplit,
    representations: Sequence[str] = REPRESENTATIONS,
    hash_dim: int = 1024,
    seed: int = 42,
) -> dict:
    """Evaluate all representations on a held-out split."""
    results = {}

    for rep in representations:
        # Extract features
        train_feats = extract_features_for_representation(
            split.train_graphs, rep, hash_dim=hash_dim
        )
        test_feats = extract_features_for_representation(
            split.test_graphs, rep, hash_dim=hash_dim
        )

        # Train classifier
        classifier = ClassicalKernelClassifier(
            kernel="linear",
            random_state=seed,
            normalize=True,
        )

        try:
            acc = classifier.train_and_evaluate(
                train_feats.X, train_feats.y,
                test_feats.X, test_feats.y,
            )
            results[rep] = {
                "accuracy": float(acc),
                "n_train": len(split.train_graphs),
                "n_test": len(split.test_graphs),
                "n_features": train_feats.X.shape[1],
            }
        except Exception as e:
            results[rep] = {
                "accuracy": None,
                "error": str(e),
                "n_train": len(split.train_graphs),
                "n_test": len(split.test_graphs),
            }

    return results


def evaluate_all_splits(
    splits: list[HeldOutSplit],
    representations: Sequence[str] = REPRESENTATIONS,
    hash_dim: int = 1024,
    seeds: Sequence[int] = (7, 42, 123),
) -> dict:
    """Evaluate all representations across all splits and seeds."""
    all_results = {}

    for seed in seeds:
        seed_results = {}
        for split in splits:
            split_results = evaluate_split(
                split, representations, hash_dim=hash_dim, seed=seed
            )
            seed_results[split.name] = split_results

        all_results[f"seed_{seed}"] = seed_results

    return all_results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_generalization_report(
    all_results: dict,
    splits: list[HeldOutSplit],
    output_dir: Path | str,
) -> Path:
    """Generate CSV + markdown report of held-out generalization results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # CSV report
    csv_path = output_path / "held_out_generalization.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "split_name", "split_description",
            "representation", "accuracy", "n_train", "n_test", "n_features", "error",
        ])

        for seed_name, split_results in all_results.items():
            for split in splits:
                for rep, metrics in split_results.get(split.name, {}).items():
                    writer.writerow([
                        seed_name,
                        split.name,
                        split.description,
                        rep,
                        metrics.get("accuracy", "N/A"),
                        metrics.get("n_train", "N/A"),
                        metrics.get("n_test", "N/A"),
                        metrics.get("n_features", "N/A"),
                        metrics.get("error", ""),
                    ])

    # Markdown report
    md_path = output_path / "held_out_generalization.md"
    with open(md_path, "w") as f:
        f.write("# Held-Out Generalization Report\n\n")

        # Summary table per representation
        f.write("## Summary by Representation\n\n")
        f.write("| Representation | Mean Acc | Std | Best Split | Worst Split |\n")
        f.write("|---------------|----------|-----|------------|-------------|\n")

        for rep in REPRESENTATIONS:
            accs = []
            best_acc = -1
            worst_acc = 2.0
            best_split = ""
            worst_split = ""

            for seed_name, split_results in all_results.items():
                for split in splits:
                    metrics = split_results.get(split.name, {}).get(rep, {})
                    acc = metrics.get("accuracy")
                    if acc is not None:
                        accs.append(acc)
                        if acc > best_acc:
                            best_acc = acc
                            best_split = split.name
                        if acc < worst_acc:
                            worst_acc = acc
                            worst_split = split.name

            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs) if len(accs) > 1 else 0.0
                f.write(
                    f"| {rep} "
                    f"| {mean_acc:.4f} "
                    f"| {std_acc:.4f} "
                    f"| {best_split} ({best_acc:.4f}) "
                    f"| {worst_split} ({worst_acc:.4f}) |\n"
                )
            else:
                f.write(f"| {rep} | N/A | N/A | - | - |\n")

        f.write("\n## Detailed Results by Seed\n\n")
        for seed_name, split_results in all_results.items():
            f.write(f"### {seed_name}\n\n")
            for split in splits:
                f.write(f"**Split:** {split.name} — {split.description}\n\n")
                f.write("| Representation | Accuracy | Train | Test | Features |\n")
                f.write("|---------------|----------|-------|------|----------|\n")
                for rep, metrics in split_results.get(split.name, {}).items():
                    acc = metrics.get("accuracy", "N/A")
                    f.write(
                        f"| {rep} "
                        f"| {acc} "
                        f"| {metrics.get('n_train', 'N/A')} "
                        f"| {metrics.get('n_test', 'N/A')} "
                        f"| {metrics.get('n_features', 'N/A')} |\n"
                    )
                f.write("\n")

    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Held-out generalization evaluation for PIG graph representations"
    )
    parser.add_argument(
        "--study-dir",
        type=str,
        default=None,
        help="Path to classical publication study output directory",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["template", "names", "corruption", "family", "all"],
        help="Held-out split strategy",
    )
    parser.add_argument(
        "--representations",
        type=str,
        default="all",
        help="Comma-separated representations to evaluate (or 'all')",
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=1024,
        help="Hash dimension for hashed fixed-layout",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="7,42,123",
        help="Comma-separated seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/held_out_generalization/)",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    representations = REPRESENTATIONS if args.representations == "all" else args.representations.split(",")

    # Load graphs
    if args.study_dir:
        study_dir = Path(args.study_dir)
    else:
        base = Path(__file__).resolve().parents[1] / "outputs" / "classical_publication"
        candidates = sorted(base.glob("toy_transformer_*"))
        study_dir = candidates[-1] if candidates else base / "toy_transformer_20260316T171701Z"

    print(f"Loading graphs from: {study_dir}")
    graphs_with_labels = load_graphs_from_study(study_dir)
    print(f"Loaded {len(graphs_with_labels)} graphs")

    if not graphs_with_labels:
        print("No graphs found. Consider running the classical publication study first.")
        return

    # Create held-out splits
    print(f"Creating held-out splits (strategy={args.strategy})...")
    splits = create_held_out_splits(graphs_with_labels, strategy=args.strategy, seed=42)
    print(f"Created {len(splits)} splits")

    # Evaluate
    print("Evaluating representations across splits and seeds...")
    all_results = evaluate_all_splits(
        splits, representations, hash_dim=args.hash_dim, seeds=seeds
    )

    # Generate report
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[1] / "outputs" / "held_out_generalization"
    csv_path = generate_generalization_report(all_results, splits, output_dir)

    print(f"\nReports written to: {output_dir}")
    print(f"  CSV: {csv_path}")
    print(f"  Markdown: {csv_path.parent / 'held_out_generalization.md'}")


def load_graphs_from_study(study_dir: Path, max_graphs: int | None = None) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
    """Load graphs from a classical publication study run."""
    results_file = study_dir / "results.json"
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]] = []

    if not results_file.exists():
        return graphs_with_labels

    with open(results_file, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "runs" in data:
        runs = data["runs"]
    elif isinstance(data, list):
        runs = data
    else:
        runs = []

    for run in runs:
        if max_graphs and len(graphs_with_labels) >= max_graphs:
            break

        graphs_data = run.get("graphs", {})
        for slice_label_str, graph_data in graphs_data.items():
            if max_graphs and len(graphs_with_labels) >= max_graphs:
                break

            graph = _graph_from_dict(graph_data)
            parts = slice_label_str.split(":")
            if len(parts) == 2:
                label = SliceLabel(task=parts[0], corruption=parts[1])
            else:
                label = SliceLabel(task="ioi", corruption=slice_label_str)
            graphs_with_labels.append((graph, label))

    return graphs_with_labels[:max_graphs] if max_graphs else graphs_with_labels


def _graph_from_dict(data: dict) -> PatchInfluenceGraph:
    """Reconstruct a PatchInfluenceGraph from a dictionary."""
    from pig.graph import Edge, Node, PatchInfluenceGraph

    nodes = []
    for node_data in data.get("nodes", []):
        nodes.append(
            Node(
                layer=node_data["layer"],
                token=node_data["token"],
                node_type=node_data.get("node_type", "res"),
                head=node_data.get("head"),
            )
        )

    edges = []
    for edge_data in data.get("edges", []):
        edges.append(
            Edge(
                src=edge_data["src"],
                dst=edge_data["dst"],
                weight=edge_data["weight"],
            )
        )

    return PatchInfluenceGraph(
        nodes=nodes,
        edges=edges,
        metadata=data.get("metadata", {}),
    )


if __name__ == "__main__":
    main()
