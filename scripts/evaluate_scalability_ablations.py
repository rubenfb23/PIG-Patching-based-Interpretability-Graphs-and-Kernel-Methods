#!/usr/bin/env python3
"""Scalability ablations for PIG graph representations.

Evaluates how representation performance depends on:
1. Top-k sparsification: k ∈ {1, 3, 5, 10}
2. Hash dimension: d ∈ {128, 256, 512, 1024}

Prioritizes toy/distilGPT2 for exploration and GPT-2 for final results.
Produces a table/artifact showing whether conclusions depend on k=5 or d=1024.
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
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib

from pig.graph import Edge, Node, PatchInfluenceGraph, build_bootstrap_slice_graphs, create_graph_builder
from pig.graph_features import (
    FixedLayoutFeatureMatrix,
    compute_fixed_layout_features_from_list,
)
from pig.kernels import ClassicalKernelClassifier
from pig.model import create_model
from pig.patching import compute_patch_effects
from pig.prompts import SliceLabel, create_ioi_dataset


# ---------------------------------------------------------------------------
# Scalability ablation parameters
# ---------------------------------------------------------------------------

TOP_K_VALUES = [1, 3, 5, 10]
HASH_DIM_VALUES = [128, 256, 512, 1024]

REPRESENTATIONS_TO_ABLATE = [
    "fixed_layout_signed",
    "hashed_fixed_layout_signed",
    "coarse_count",
]


# ---------------------------------------------------------------------------
# Top-k ablation
# ---------------------------------------------------------------------------

def run_topk_ablation(
    model_name: str = "toy_transformer",
    k_values: Sequence[int] = TOP_K_VALUES,
    seeds: Sequence[int] = (7, 42, 123),
    num_examples: int = 100,
    output_dir: Path | str = "outputs/scalability_ablations",
    continue_on_error: bool = False,
) -> dict:
    """Run top-k ablation: evaluate representations at different k values.

    For each k value:
    1. Build graphs with that k
    2. Extract features for each representation
    3. Evaluate with linear SVM
    4. Report accuracy per seed and mean/std
    """
    output_path = Path(output_dir) / "topk_ablation"
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for k in k_values:
        print(f"\n--- Top-k ablation: k={k} ---")
        seed_results = {}

        for seed in seeds:
            print(f"  Seed {seed}...")
            try:
                seed_accs = _run_topk_seed(
                    model_name, k, seed, num_examples, REPRESENTATIONS_TO_ABLATE
                )
                seed_results[seed] = seed_accs
                results[f"k={k}"] = seed_results
            except Exception as e:
                if continue_on_error:
                    print(f"  Error at k={k}, seed={seed}: {e}")
                    seed_results[seed] = {}
                    results[f"k={k}"] = seed_results
                else:
                    raise

    # Generate reports
    _generate_topk_reports(results, output_path)

    return results


def _run_topk_seed(
    model_name: str,
    k: int,
    seed: int,
    num_examples: int,
    representations: Sequence[str],
) -> dict:
    """Run top-k evaluation for a single seed."""
    # Create model
    model = create_model(model_name)

    # Create IOI dataset
    dataset = create_ioi_dataset(
        n_examples=num_examples,
        seed=seed,
    )

    # Compute patch effects
    patch_effects = compute_patch_effects(
        model=model,
        prompt_pairs=dataset,
        node_types=["res"],
    )

    # Build graphs with specific k
    graph_builder = create_graph_builder(
        "correlation_topk",
        k=k,
        enforce_direction=True,
    )

    graphs_with_labels = build_bootstrap_slice_graphs(
        patch_effects,
        graph_builder,
        graphs_per_slice=32,
        sample_fraction=0.75,
        min_examples=3,
        seed=seed,
    )

    # Extract features for each representation
    results = {}
    for rep in representations:
        feats = compute_fixed_layout_features_from_list(graphs_with_labels)

        if rep == "fixed_layout_signed":
            X = np.sign(feats.to_matrix()).astype(np.float64)
        elif rep == "hashed_fixed_layout_signed":
            X = _hash_features(feats.to_matrix(), 1024)
        elif rep == "coarse_count":
            X = _coarse_count_from_matrix(feats.to_matrix(), feats.feature_names)
        else:
            X = feats.to_matrix().astype(np.float64)

        label_map = {"name_swap": 0, "abba": 1}
        y = np.array([label_map.get(str(l).split(":")[-1], 0)
                      for l in feats.slice_labels], dtype=np.int32)

        # Proper train/test split (80/20, stratified) instead of leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )

        classifier = ClassicalKernelClassifier(
            kernel="linear",
            random_state=seed,
            normalize=True,
        )
        acc = classifier.train_and_evaluate(X_train, y_train, X_test, y_test)
        results[rep] = float(acc)

    return results


# ---------------------------------------------------------------------------
# Hash dimension ablation
# ---------------------------------------------------------------------------

def run_hash_dim_ablation(
    model_name: str = "toy_transformer",
    dim_values: Sequence[int] = HASH_DIM_VALUES,
    seeds: Sequence[int] = (7, 42, 123),
    num_examples: int = 100,
    output_dir: Path | str = "outputs/scalability_ablations",
    continue_on_error: bool = False,
) -> dict:
    """Run hash dimension ablation: evaluate hashed fixed-layout at different dims.

    For each dimension:
    1. Build graphs (k=5)
    2. Extract hashed features with that dimension
    3. Evaluate with linear SVM
    4. Report accuracy per seed and mean/std
    """
    output_path = Path(output_dir) / "hash_dim_ablation"
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for dim in dim_values:
        print(f"\n--- Hash dim ablation: d={dim} ---")
        seed_results = {}

        for seed in seeds:
            print(f"  Seed {seed}...")
            try:
                seed_accs = _run_hash_dim_seed(
                    model_name, dim, seed, num_examples
                )
                seed_results[seed] = seed_accs
                results[f"d={dim}"] = seed_results
            except Exception as e:
                if continue_on_error:
                    print(f"  Error at d={dim}, seed={seed}: {e}")
                    seed_results[seed] = {}
                    results[f"d={dim}"] = seed_results
                else:
                    raise

    # Generate reports
    _generate_hash_dim_reports(results, output_path)

    return results


def _run_hash_dim_seed(
    model_name: str,
    dim: int,
    seed: int,
    num_examples: int,
) -> dict:
    """Run hash dimension evaluation for a single seed."""
    model = create_model(model_name)
    dataset = create_ioi_dataset(
        n_examples=num_examples,
        seed=seed,
    )
    patch_effects = compute_patch_effects(
        model=model,
        prompt_pairs=dataset,
        node_types=["res"],
    )

    graph_builder = create_graph_builder(
        "correlation_topk",
        k=5,
        enforce_direction=True,
    )

    graphs_with_labels = build_bootstrap_slice_graphs(
        patch_effects,
        graph_builder,
        graphs_per_slice=32,
        sample_fraction=0.75,
        min_examples=3,
        seed=seed,
    )

    feats = compute_fixed_layout_features_from_list(graphs_with_labels)
    X = _hash_features(feats.to_matrix(), dim)

    label_map = {"name_swap": 0, "abba": 1}
    y = np.array([label_map.get(str(l).split(":")[-1], 0)
                  for l in feats.slice_labels], dtype=np.int32)

    # Proper train/test split (80/20, stratified) instead of leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    classifier = ClassicalKernelClassifier(
        kernel="linear",
        random_state=seed,
        normalize=True,
    )
    acc = classifier.train_and_evaluate(X_train, y_train, X_test, y_test)

    return {"hashed_fixed_layout_signed": float(acc)}


# ---------------------------------------------------------------------------
# Combined ablation
# ---------------------------------------------------------------------------

def run_combined_ablation(
    model_name: str = "toy_transformer",
    k_values: Sequence[int] = TOP_K_VALUES,
    dim_values: Sequence[int] = HASH_DIM_VALUES,
    seeds: Sequence[int] = (7, 42, 123),
    num_examples: int = 100,
    output_dir: Path | str = "outputs/scalability_ablations",
    continue_on_error: bool = False,
) -> dict:
    """Run combined top-k x hash-dim ablation.

    Produces a grid of results showing how accuracy depends on both k and d.
    """
    output_path = Path(output_dir) / "combined_ablation"
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for k in k_values:
        for dim in dim_values:
            print(f"\n--- Combined: k={k}, d={dim} ---")
            seed_results = {}

            for seed in seeds:
                print(f"  Seed {seed}...")
                try:
                    seed_accs = _run_combined_seed(
                        model_name, k, dim, seed, num_examples
                    )
                    seed_results[seed] = seed_accs
                    results[f"k={k}_d={dim}"] = seed_results
                except Exception as e:
                    if continue_on_error:
                        print(f"  Error: {e}")
                        seed_results[seed] = {}
                        results[f"k={k}_d={dim}"] = seed_results
                    else:
                        raise

    _generate_combined_reports(results, output_path)

    return results


def _run_combined_seed(
    model_name: str,
    k: int,
    dim: int,
    seed: int,
    num_examples: int,
) -> dict:
    """Run combined ablation for a single seed."""
    model = create_model(model_name)
    dataset = create_ioi_dataset(
        n_examples=num_examples,
        seed=seed,
    )
    patch_effects = compute_patch_effects(
        model=model,
        prompt_pairs=dataset,
        node_types=["res"],
    )

    graph_builder = create_graph_builder(
        "correlation_topk",
        k=k,
        enforce_direction=True,
    )

    graphs_with_labels = build_bootstrap_slice_graphs(
        patch_effects,
        graph_builder,
        graphs_per_slice=32,
        sample_fraction=0.75,
        min_examples=3,
        seed=seed,
    )

    feats = compute_fixed_layout_features_from_list(graphs_with_labels)
    X = _hash_features(feats.to_matrix(), dim)

    label_map = {"name_swap": 0, "abba": 1}
    y = np.array([label_map.get(str(l).split(":")[-1], 0)
                  for l in feats.slice_labels], dtype=np.int32)

    # Proper train/test split (80/20, stratified) instead of leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    classifier = ClassicalKernelClassifier(
        kernel="linear",
        random_state=seed,
        normalize=True,
    )
    acc = classifier.train_and_evaluate(X_train, y_train, X_test, y_test)

    return {"hashed_fixed_layout_signed": float(acc)}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_topk_reports(results: dict, output_path: Path) -> None:
    """Generate CSV + markdown reports for top-k ablation."""
    # CSV
    csv_path = output_path / "topk_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "representation", "seed", "accuracy"])
        for k_str, seed_results in results.items():
            for seed, rep_results in seed_results.items():
                for rep, acc in rep_results.items():
                    k_val = k_str.replace("k=", "")
                    writer.writerow([k_val, rep, seed, acc])

    # Markdown
    md_path = output_path / "topk_ablation.md"
    with open(md_path, "w") as f:
        f.write("# Top-k Ablation Report\n\n")
        f.write("How classification accuracy depends on top-k sparsification.\n\n")
        f.write("| k | Representation | Mean Acc | Std |\n")
        f.write("|---|---------------|----------|---|\n")

        for rep in REPRESENTATIONS_TO_ABLATE:
            accs = []
            for k_str, seed_results in results.items():
                for seed, rep_results in seed_results.items():
                    if rep in rep_results:
                        accs.append(rep_results[rep])

            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs) if len(accs) > 1 else 0.0
                f.write(f"| - | {rep} | {mean_acc:.4f} | {std_acc:.4f} |\n")

        f.write("\n## Detailed Results\n\n")
        f.write("| k | Seed | Representation | Accuracy |\n")
        f.write("|---|------|---------------|----------|\n")
        for k_str, seed_results in results.items():
            for seed, rep_results in seed_results.items():
                for rep, acc in rep_results.items():
                    f.write(f"| {k_str.replace('k=', '')} | {seed} | {rep} | {acc:.4f} |\n")


def _generate_hash_dim_reports(results: dict, output_path: Path) -> None:
    """Generate CSV + markdown reports for hash dimension ablation."""
    csv_path = output_path / "hash_dim_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dim", "seed", "accuracy"])
        for dim_str, seed_results in results.items():
            for seed, rep_results in seed_results.items():
                dim_val = dim_str.replace("d=", "")
                for rep, acc in rep_results.items():
                    writer.writerow([dim_val, seed, acc])

    md_path = output_path / "hash_dim_ablation.md"
    with open(md_path, "w") as f:
        f.write("# Hash Dimension Ablation Report\n\n")
        f.write("How classification accuracy depends on hash dimension for hashed fixed-layout.\n\n")
        f.write("| Dim | Mean Acc | Std |\n")
        f.write("|---|----------|---|\n")

        for dim_str, seed_results in results.items():
            accs = []
            for seed, rep_results in seed_results.items():
                for rep, acc in rep_results.items():
                    accs.append(acc)

            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs) if len(accs) > 1 else 0.0
                dim_val = dim_str.replace("d=", "")
                f.write(f"| {dim_val} | {mean_acc:.4f} | {std_acc:.4f} |\n")


def _generate_combined_reports(results: dict, output_path: Path) -> None:
    """Generate CSV + markdown reports for combined ablation."""
    csv_path = output_path / "combined_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "dim", "seed", "accuracy"])
        for kd_str, seed_results in results.items():
            k_val, dim_val = kd_str.replace("k=", "").replace("_d=", ",").split(",")
            for seed, rep_results in seed_results.items():
                for rep, acc in rep_results.items():
                    writer.writerow([k_val, dim_val, seed, acc])

    md_path = output_path / "combined_ablation.md"
    with open(md_path, "w") as f:
        f.write("# Combined Top-k x Hash-Dim Ablation Report\n\n")
        f.write("Grid of accuracy values across top-k and hash dimension.\n\n")

        # Pivot table
        f.write("| k \\ d | 128 | 256 | 512 | 1024 |\n")
        f.write("|------|---|---|---|-----|\n")

        for k in TOP_K_VALUES:
            row = []
            for dim in HASH_DIM_VALUES:
                kd_str = f"k={k}_d={dim}"
                accs = []
                seed_results = results.get(kd_str, {})
                for seed, rep_results in seed_results.items():
                    for rep, acc in rep_results.items():
                        accs.append(acc)
                mean_acc = np.mean(accs) if accs else 0.0
                row.append(f"{mean_acc:.4f}")
            f.write(f"| {k} | {' | '.join(row)} |\n")

        f.write("\n## Interpretation\n\n")
        f.write("- If accuracy is stable across k values, conclusions don't depend on k=5\n")
        f.write("- If accuracy saturates at d=512, d=1024 is unnecessary\n")
        f.write("- Look for interaction effects: does optimal d depend on k?\n")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _hash_features(
    matrix: NDArray[np.float64],
    dim: int = 1024,
) -> NDArray[np.float64]:
    """Hash fixed-layout features to fixed dimension."""
    n_graphs, n_features = matrix.shape
    hashed = np.zeros((n_graphs, dim), dtype=np.float64)
    for j in range(n_features):
        h = hash((j, dim)) % dim
        hashed[:, h] += matrix[:, j]
    return hashed


def _coarse_count_from_matrix(
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
        delta_layer = abs(dst_layer - src_layer)
        layer_bucket = min(delta_layer // 2, 5)
        bucket_key = f"l{layer_bucket}"
        buckets[bucket_key] += matrix[0, j]

    bucket_keys = sorted(buckets.keys())
    X = np.zeros((n_graphs, len(bucket_keys)), dtype=np.float64)
    for i, key in enumerate(bucket_keys):
        X[:, i] = buckets[key]

    return X


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scalability ablations for PIG graph representations"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="toy_transformer",
        help="Model name (toy_transformer, gpt2)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=["topk", "hash_dim", "all"],
        help="Which ablation to run",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5,10",
        help="Comma-separated k values",
    )
    parser.add_argument(
        "--dim-values",
        type=str,
        default="128,256,512,1024",
        help="Comma-separated hash dimension values",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="7,42,123",
        help="Comma-separated seeds",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of prompt pairs per corruption",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/scalability_ablations/)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running even if one configuration fails",
    )
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    dim_values = [int(x) for x in args.dim_values.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[1] / "outputs" / "scalability_ablations"

    if args.ablation in ("topk", "all"):
        print("=" * 60)
        print("Running top-k ablation...")
        print("=" * 60)
        run_topk_ablation(
            model_name=args.model_name,
            k_values=k_values,
            seeds=seeds,
            num_examples=args.num_examples,
            output_dir=output_dir,
            continue_on_error=args.continue_on_error,
        )

    if args.ablation in ("hash_dim", "all"):
        print("\n" + "=" * 60)
        print("Running hash dimension ablation...")
        print("=" * 60)
        run_hash_dim_ablation(
            model_name=args.model_name,
            dim_values=dim_values,
            seeds=seeds,
            num_examples=args.num_examples,
            output_dir=output_dir,
            continue_on_error=args.continue_on_error,
        )

    if args.ablation == "all":
        print("\n" + "=" * 60)
        print("Running combined ablation...")
        print("=" * 60)
        run_combined_ablation(
            model_name=args.model_name,
            k_values=k_values,
            dim_values=dim_values,
            seeds=seeds,
            num_examples=args.num_examples,
            output_dir=output_dir,
            continue_on_error=args.continue_on_error,
        )

    print(f"\nAll reports written to: {output_dir}")


if __name__ == "__main__":
    main()
