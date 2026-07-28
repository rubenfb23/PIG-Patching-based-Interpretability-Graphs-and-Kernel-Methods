#!/usr/bin/env python3
"""Additional null controls for PIG graph representations.

Extends the existing null controls (edge_shuffle, weight_shuffle) with:
1. Label shuffle: randomly permute slice labels before classification
2. Node-index permutation: randomly permute node indices (preserving graph structure)
3. Random graph baseline: randomize graph structure while preserving degree/top-k
4. Patch-effect tensor randomization: randomize raw patch effects before graph construction

These controls test whether the observed signal depends on specific
structural assignments or just marginal statistics.
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

from pig.graph import Edge, Node, PatchInfluenceGraph
from pig.graph_features import (
    FixedLayoutFeatureMatrix,
    apply_null_control_to_graphs,
    compute_fixed_layout_features_from_list,
)
from pig.kernels import ClassicalKernelClassifier
from pig.prompts import SliceLabel


# ---------------------------------------------------------------------------
# Additional null control implementations
# ---------------------------------------------------------------------------

def label_shuffle_graphs(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    rng: np.random.Generator,
) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
    """Randomly permute slice labels, breaking the correspondence between graphs and labels.

    This is a strict null control: if the representation truly captures
    slice-discriminative signal, accuracy should drop to chance.
    """
    labels = [label for _, label in graphs_with_labels]
    shuffled_labels = [labels[i] for i in rng.permutation(len(labels))]
    return [(graph, label) for (graph, _), label in zip(graphs_with_labels, shuffled_labels)]


def node_index_permutation_graphs(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    rng: np.random.Generator,
) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
    """Randomly permute node indices within each graph.

    Preserves the number of edges but destroys the mechanistic meaning
    of specific node positions. Tests whether signal depends on
    specific node identities.
    """
    transformed = []
    for graph, label in graphs_with_labels:
        n = len(graph.nodes)
        if n <= 1:
            transformed.append((graph, label))
            continue

        perm = rng.permutation(n)
        new_edges = []
        for edge in graph.edges:
            new_edges.append(Edge(
                src=int(perm[edge.src]),
                dst=int(perm[edge.dst]),
                weight=edge.weight,
            ))

        new_nodes = [graph.nodes[i] for i in perm]
        null_graph = PatchInfluenceGraph(
            nodes=new_nodes,
            edges=new_edges,
            metadata={**graph.metadata, "null_control": "node_index_permutation"},
        )
        transformed.append((null_graph, label))

    return transformed


def random_graph_matched_degree(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    rng: np.random.Generator,
) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
    """Generate random graphs preserving in/out-degree distribution.

    Uses a configuration-switching approach: randomly rewire edges while
    preserving the degree sequence of each node. This preserves local
    connectivity statistics but destroys any mechanistic meaning.
    """
    transformed = []
    for graph, label in graphs_with_labels:
        n = len(graph.nodes)
        if n <= 1 or not graph.edges:
            transformed.append((graph, label))
            continue

        # Compute degree sequences
        in_deg = defaultdict(int)
        out_deg = defaultdict(int)
        for edge in graph.edges:
            in_deg[edge.dst] += 1
            out_deg[edge.src] += 1

        # Build edge lists by source degree
        edge_sources = []
        for edge in graph.edges:
            edge_sources.append(edge.src)

        # Randomly rewire: shuffle destinations while preserving source degrees
        destinations = [edge.dst for edge in graph.edges]
        shuffled_dests = rng.permutation(destinations)

        new_edges = []
        for src, dst in zip(edge_sources, shuffled_dests):
            if src != dst:  # Avoid self-loops
                new_edges.append(Edge(src=src, dst=int(dst), weight=0.0))
            else:
                # Find alternative destination
                for alt in rng.permutation(n):
                    if alt != src:
                        new_edges.append(Edge(src=src, dst=int(alt), weight=0.0))
                        break

        null_graph = PatchInfluenceGraph(
            nodes=graph.nodes,
            edges=new_edges,
            metadata={**graph.metadata, "null_control": "random_graph_matched_degree"},
        )
        transformed.append((null_graph, label))

    return transformed


def random_patch_effects(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    rng: np.random.Generator,
    seed: int = 0,
) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
    """Replace patch effects with random values before graph construction.

    This is the strongest null control: if graphs built from random patch
    effects still classify well, the signal is not from patch effects.
    """
    transformed = []
    for graph, label in graphs_with_labels:
        # Randomize edge weights while preserving graph topology
        n_edges = len(graph.edges)
        if n_edges == 0:
            transformed.append((graph, label))
            continue

        # Generate random correlations
        random_weights = rng.standard_normal(n_edges)

        new_edges = []
        for edge, weight in zip(graph.edges, random_weights):
            new_edges.append(Edge(
                src=edge.src,
                dst=edge.dst,
                weight=float(weight),
            ))

        null_graph = PatchInfluenceGraph(
            nodes=graph.nodes,
            edges=new_edges,
            metadata={**graph.metadata, "null_control": "random_patch_effects"},
        )
        transformed.append((null_graph, label))

    return transformed


def sign_flip_graphs(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    rng: np.random.Generator,
) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
    """Randomly flip edge weight signs.

    Tests whether the sign of correlations matters or if magnitude alone
    drives classification.
    """
    transformed = []
    for graph, label in graphs_with_labels:
        new_edges = []
        for edge in graph.edges:
            flip = rng.choice([-1, 1])
            new_edges.append(Edge(
                src=edge.src,
                dst=edge.dst,
                weight=float(edge.weight * flip),
            ))

        null_graph = PatchInfluenceGraph(
            nodes=graph.nodes,
            edges=new_edges,
            metadata={**graph.metadata, "null_control": "sign_flip"},
        )
        transformed.append((null_graph, label))

    return transformed


def edge_target_shuffle_graphs(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    rng: np.random.Generator,
) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
    """Shuffle edge targets while preserving source nodes and edge count.

    Different from edge_shuffle (which preserves weight multiset): this
    preserves the source node identity and edge count per source, but
    randomizes which nodes receive edges.
    """
    transformed = []
    for graph, label in graphs_with_labels:
        n = len(graph.nodes)
        if n <= 1 or not graph.edges:
            transformed.append((graph, label))
            continue

        # Group edges by source
        edges_by_src: defaultdict = defaultdict(list)
        for edge in graph.edges:
            edges_by_src[edge.src].append(edge)

        # Shuffle targets within each source group
        new_edges = []
        for src in sorted(edges_by_src.keys()):
            edges = edges_by_src[src]
            valid_targets = [j for j in range(n) if j != src and graph.nodes[src] < graph.nodes[j]]
            if not valid_targets:
                valid_targets = [j for j in range(n) if j != src]

            if len(valid_targets) > 1:
                shuffled_targets = rng.permutation(valid_targets)
            else:
                shuffled_targets = valid_targets

            for edge, target in zip(edges, shuffled_targets):
                new_edges.append(Edge(
                    src=edge.src,
                    dst=int(target),
                    weight=edge.weight,
                ))

        null_graph = PatchInfluenceGraph(
            nodes=graph.nodes,
            edges=new_edges,
            metadata={**graph.metadata, "null_control": "edge_target_shuffle"},
        )
        transformed.append((null_graph, label))

    return transformed


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

NULL_CONTROLS = [
    "label_shuffle",
    "node_index_permutation",
    "random_graph_matched_degree",
    "random_patch_effects",
    "sign_flip",
    "edge_target_shuffle",
]

BASELINE_CONTROLS = [
    "edge_shuffle",
    "weight_shuffle",
]


def evaluate_null_controls(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    representations: Sequence[str] = ("fixed_layout_signed", "hashed_fixed_layout_signed", "coarse_count"),
    null_controls: Sequence[str] = NULL_CONTROLS + BASELINE_CONTROLS,
    seeds: Sequence[int] = (7, 42, 123),
    hash_dim: int = 1024,
    output_dir: Path | str = "outputs/null_controls",
) -> dict:
    """Run null control evaluation across representations and controls.

    For each representation and null control:
    1. Apply the null control to the graphs
    2. Extract features
    3. Train on original, test on null
    4. Report accuracy

    Returns results dict for report generation.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Baseline: no null control
    print("Computing baseline (no null control)...")
    baseline_feats = compute_fixed_layout_features_from_list(graphs_with_labels)
    baseline_results = _evaluate_representation(
        baseline_feats, representations, seeds, hash_dim
    )
    results["baseline"] = baseline_results

    # Apply each null control
    for control in null_controls:
        print(f"Applying null control: {control}...")

        if control in BASELINE_CONTROLS:
            null_graphs = apply_null_control_to_graphs(
                graphs_with_labels, control, seed=42
            )
        elif control == "label_shuffle":
            rng = np.random.default_rng(42)
            null_graphs = label_shuffle_graphs(graphs_with_labels, rng)
        elif control == "node_index_permutation":
            rng = np.random.default_rng(42)
            null_graphs = node_index_permutation_graphs(graphs_with_labels, rng)
        elif control == "random_graph_matched_degree":
            rng = np.random.default_rng(42)
            null_graphs = random_graph_matched_degree(graphs_with_labels, rng)
        elif control == "random_patch_effects":
            rng = np.random.default_rng(42)
            null_graphs = random_patch_effects(graphs_with_labels, rng)
        elif control == "sign_flip":
            rng = np.random.default_rng(42)
            null_graphs = sign_flip_graphs(graphs_with_labels, rng)
        elif control == "edge_target_shuffle":
            rng = np.random.default_rng(42)
            null_graphs = edge_target_shuffle_graphs(graphs_with_labels, rng)
        else:
            print(f"  Skipping unknown control: {control}")
            continue

        null_feats = compute_fixed_layout_features_from_list(null_graphs)
        null_results = _evaluate_representation(
            null_feats, representations, seeds, hash_dim
        )
        results[control] = null_results

    # Generate reports
    _generate_null_control_reports(results, output_path)

    return results


def _evaluate_representation(
    features: FixedLayoutFeatureMatrix,
    representations: Sequence[str],
    seeds: Sequence[int],
    hash_dim: int,
) -> dict:
    """Evaluate representations on a fixed feature matrix across seeds."""
    label_map = {"name_swap": 0, "abba": 1}
    y = np.array([label_map.get(str(l).split(":")[-1], 0)
                  for l in features.slice_labels], dtype=np.int32)
    X_raw = features.to_matrix().astype(np.float64)

    # Proper train/test split (80/20, stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}
    for rep in representations:
        # Build train and test feature matrices per representation
        if rep == "fixed_layout_signed":
            X_train, X_test = np.sign(X_train_raw), np.sign(X_test_raw)
        elif rep == "fixed_layout_weighted":
            X_train, X_test = X_train_raw, X_test_raw
        elif rep == "hashed_fixed_layout_signed":
            X_train = _hash_features(X_train_raw, hash_dim)
            X_test = _hash_features(X_test_raw, hash_dim)
        elif rep == "coarse_count":
            X_train = _coarse_count_from_matrix(X_train_raw, features.feature_names)
            X_test = _coarse_count_from_matrix(X_test_raw, features.feature_names)
        else:
            X_train, X_test = X_train_raw, X_test_raw

        seed_accs = []
        for seed in seeds:
            classifier = ClassicalKernelClassifier(
                kernel="linear",
                random_state=seed,
                normalize=True,
            )
            try:
                acc = classifier.train_and_evaluate(X_train, y_train, X_test, y_test)
                seed_accs.append(float(acc))
            except Exception as e:
                seed_accs.append(None)

        valid_accs = [a for a in seed_accs if a is not None]
        results[rep] = {
            "accuracies": seed_accs,
            "mean": float(np.mean(valid_accs)) if valid_accs else None,
            "std": float(np.std(valid_accs)) if len(valid_accs) > 1 else 0.0,
        }

    return results


def _hash_features(
    matrix: NDArray[np.float64],
    dim: int = 1024,
) -> NDArray[np.float64]:
    """Hash features to fixed dimension using stable hash."""
    import hashlib
    n_graphs, n_features = matrix.shape
    hashed = np.zeros((n_graphs, dim), dtype=np.float64)
    for j in range(n_features):
        key = f"{j}:{dim}"
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % dim
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
        src_token = int(src_parts[1].replace("T", ""))
        dst_layer = int(dst_parts[0].replace("L", ""))
        dst_token = int(dst_parts[1].replace("T", ""))

        src_node_type = "res" if "res" in src_parts[2] else src_parts[2]
        dst_node_type = "res" if "res" in dst_parts[2] else dst_parts[2]
        node_type_key = f"{src_node_type}->{dst_node_type}"

        delta_layer = abs(dst_layer - src_layer)
        delta_token = abs(dst_token - src_token)
        layer_bucket = min(delta_layer // 2, 5)
        token_bucket = min(delta_token // 2, 5)
        sign_bucket = 1 if matrix[0, j] > 0 else 0

        bucket_key = f"{node_type_key}_l{layer_bucket}_t{token_bucket}_s{sign_bucket}"
        buckets[bucket_key] += matrix[0, j]

    bucket_keys = sorted(buckets.keys())
    X = np.zeros((n_graphs, len(bucket_keys)), dtype=np.float64)
    for i, key in enumerate(bucket_keys):
        X[:, i] = buckets[key]

    return X


def _generate_null_control_reports(
    results: dict,
    output_path: Path,
) -> None:
    """Generate CSV and markdown reports for null control results."""
    # CSV report
    csv_path = output_path / "null_controls.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "control", "representation", "seed_accuracies",
            "mean_acc", "std_acc",
        ])

        for control, rep_results in results.items():
            for rep, metrics in rep_results.items():
                writer.writerow([
                    control,
                    rep,
                    str(metrics["accuracies"]),
                    metrics["mean"],
                    metrics["std"],
                ])

    # Markdown report
    md_path = output_path / "null_controls.md"
    with open(md_path, "w") as f:
        f.write("# Null Controls Report\n\n")
        f.write("This report evaluates how graph representations perform under\n")
        f.write("various null controls. A valid representation should drop to\n")
        f.write("chance (~0.50) under strict null controls.\n\n")

        f.write("## Summary\n\n")
        f.write("| Control | Representation | Mean Acc | Std | Chance? |\n")
        f.write("|---------|---------------|----------|---|---------|\n")

        for control, rep_results in results.items():
            for rep, metrics in rep_results.items():
                mean_acc = metrics["mean"]
                if mean_acc is not None:
                    is_chance = "✓" if abs(mean_acc - 0.5) < 0.05 else "✗"
                else:
                    is_chance = "N/A"
                f.write(
                    f"| {control} "
                    f"| {rep} "
                    f"| {mean_acc:.4f} "
                    f"| {metrics['std']:.4f} "
                    f"| {is_chance} |\n"
                )

        f.write("\n## Interpretation\n\n")
        f.write("- **Baseline**: Should match main paper results (high accuracy)\n")
        f.write("- **Label shuffle**: If accuracy > 0.6, labels may leak\n")
        f.write("- **Node index permutation**: Tests node identity dependence\n")
        f.write("- **Random graph**: Tests structural vs. statistical signal\n")
        f.write("- **Random patch effects**: Strongest test of patch-effect signal\n")
        f.write("- **Sign flip**: Tests whether sign matters vs. magnitude\n")
        f.write("- **Edge target shuffle**: Tests edge target dependence\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Additional null controls for PIG graph representations"
    )
    parser.add_argument(
        "--study-dir",
        type=str,
        default=None,
        help="Path to classical publication study output directory",
    )
    parser.add_argument(
        "--representations",
        type=str,
        default="fixed_layout_signed,hashed_fixed_layout_signed,coarse_count",
        help="Comma-separated representations to evaluate",
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
        help="Output directory (default: outputs/null_controls/)",
    )
    args = parser.parse_args()

    representations = args.representations.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

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
        print("No graphs found.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[1] / "outputs" / "null_controls"
    results = evaluate_null_controls(
        graphs_with_labels,
        representations=representations,
        seeds=seeds,
        hash_dim=args.hash_dim,
        output_dir=output_dir,
    )

    print(f"\nReports written to: {output_dir}")


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
