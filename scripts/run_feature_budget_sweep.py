#!/usr/bin/env python3
"""Sweep compression budgets against the fixed-layout signed baseline.

This experiment asks how many compressed features are needed to recover a fixed
fraction of the exact fixed-layout signal. It reuses cached patch-effect tensors
from the paper-decision runs, builds the same example-disjoint bootstrap graphs,
and evaluates each representation with the same linear SVM protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pig.graph import create_graph_builder
from pig.patching import PatchEffectDataset, PatchEffectTensor

from scripts.run_paper_decision_study import (
    MatrixBundle,
    _build_bootstrap_graphs,
    _fit_eval_svm,
    _fixed_layout_matrix,
    _graphlet_matrix,
    _hashed_fixed_layout_matrix,
    _spectral_matrix,
    _split_dataset_by_slice,
)


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_bin_pairs(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            value = int(part)
            pairs.append((value, value))
            continue
        left, right = part.split("x", 1)
        pairs.append((int(left), int(right)))
    return pairs


def _load_dataset(cache_dir: Path) -> PatchEffectDataset:
    dataset = PatchEffectDataset()
    for path in sorted(cache_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            tensor = PatchEffectTensor.from_dict(json.load(handle))
        dataset.add(tensor)
    if not len(dataset):
        raise ValueError(f"No cached patch-effect tensors found in {cache_dir}")
    return dataset


def _eval_pair(train: MatrixBundle, test: MatrixBundle, *, seed: int) -> float:
    return _fit_eval_svm(train, test, kernel="linear", seed=seed)["accuracy"]


def _signed_delta_bin(delta: int, *, bins: int, max_abs: int) -> str:
    if delta == 0:
        return "z0"
    sign = "p" if delta > 0 else "n"
    magnitude = min(abs(delta), max(max_abs, 1))
    width = max_abs / max(bins, 1)
    bucket = min(int((magnitude - 1e-9) / max(width, 1e-9)), bins - 1)
    return f"{sign}{bucket}"


def _position_bin(value: int, *, bins: int, max_value: int) -> int:
    if bins <= 1 or max_value <= 0:
        return 0
    return min(int(value * bins / (max_value + 1)), bins - 1)


def _coarse_budget_key(graph, edge, *, layer_bins: int, token_bins: int) -> str:
    src = graph.nodes[edge.src]
    dst = graph.nodes[edge.dst]
    sign = "pos" if edge.weight > 0 else "neg" if edge.weight < 0 else "zero"
    layer_delta = _signed_delta_bin(
        dst.layer - src.layer,
        bins=layer_bins,
        max_abs=max(graph.num_layers - 1, 1),
    )
    token_delta = _signed_delta_bin(
        dst.token - src.token,
        bins=token_bins,
        max_abs=max(graph.num_tokens - 1, 1),
    )
    return "|".join([src.node_type, dst.node_type, layer_delta, token_delta, sign])


def _coarse_budget_matrix(
    graphs_with_labels,
    *,
    layer_bins: int,
    token_bins: int,
    feature_names: list[str] | None = None,
) -> MatrixBundle:
    labels = [label for _, label in graphs_with_labels]
    if feature_names is None:
        features = {
            _coarse_budget_key(
                graph,
                edge,
                layer_bins=layer_bins,
                token_bins=token_bins,
            )
            for graph, _ in graphs_with_labels
            for edge in graph.edges
        }
        feature_names = sorted(features)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    matrix = np.zeros((len(graphs_with_labels), len(feature_names)), dtype=np.float32)
    for row_idx, (graph, _) in enumerate(graphs_with_labels):
        for edge in graph.edges:
            key = _coarse_budget_key(
                graph,
                edge,
                layer_bins=layer_bins,
                token_bins=token_bins,
            )
            col_idx = feature_index.get(key)
            if col_idx is not None:
                matrix[row_idx, col_idx] += np.float32(1.0)
    return MatrixBundle(X=matrix, y=labels, feature_names=feature_names)


def _dense_binary_signed(graph) -> tuple[np.ndarray, np.ndarray]:
    binary = np.zeros((graph.num_nodes, graph.num_nodes), dtype=np.float64)
    signed = np.zeros((graph.num_nodes, graph.num_nodes), dtype=np.float64)
    for edge in graph.edges:
        binary[edge.src, edge.dst] = 1.0
        binary[edge.dst, edge.src] = 1.0
        sign = 1.0 if edge.weight > 0 else -1.0 if edge.weight < 0 else 0.0
        signed[edge.src, edge.dst] = sign
        signed[edge.dst, edge.src] = sign
    return binary, signed


def _graphlet_orbit_features(
    graph,
    *,
    layer_bins: int,
    token_bins: int,
) -> np.ndarray:
    binary, signed = _dense_binary_signed(graph)
    degree = binary.sum(axis=1)
    positive_degree = (signed > 0).sum(axis=1).astype(np.float64)
    negative_degree = (signed < 0).sum(axis=1).astype(np.float64)
    wedge_center = degree * (degree - 1.0) / 2.0

    num_groups = layer_bins * token_bins
    stats_per_group = 6
    features = np.zeros(num_groups * stats_per_group, dtype=np.float32)
    counts = np.zeros(num_groups, dtype=np.float64)

    for node_idx, node in enumerate(graph.nodes):
        layer_bin = _position_bin(
            node.layer,
            bins=layer_bins,
            max_value=max(graph.num_layers - 1, 0),
        )
        token_bin = _position_bin(
            node.token,
            bins=token_bins,
            max_value=max(graph.num_tokens - 1, 0),
        )
        group = layer_bin * token_bins + token_bin
        offset = group * stats_per_group
        counts[group] += 1.0
        features[offset + 0] += np.float32(1.0 / max(graph.num_nodes, 1))
        features[offset + 1] += np.float32(degree[node_idx])
        features[offset + 2] += np.float32(wedge_center[node_idx])
        features[offset + 3] += np.float32(positive_degree[node_idx])
        features[offset + 4] += np.float32(negative_degree[node_idx])
        features[offset + 5] += np.float32(degree[node_idx] ** 2)

    for group, count in enumerate(counts):
        if count <= 0:
            continue
        offset = group * stats_per_group
        features[offset + 1 : offset + stats_per_group] /= np.float32(count)
    return features


def _graphlet_orbit_matrix(
    graphs_with_labels,
    *,
    layer_bins: int,
    token_bins: int,
) -> MatrixBundle:
    matrix = np.stack(
        [
            _graphlet_orbit_features(
                graph,
                layer_bins=layer_bins,
                token_bins=token_bins,
            )
            for graph, _ in graphs_with_labels
        ],
    )
    labels = [label for _, label in graphs_with_labels]
    stat_names = [
        "node_count",
        "degree",
        "wedge_center",
        "positive_degree",
        "negative_degree",
        "degree_sq",
    ]
    feature_names = [
        f"L{layer_idx}:T{token_idx}:{name}"
        for layer_idx in range(layer_bins)
        for token_idx in range(token_bins)
        for name in stat_names
    ]
    return MatrixBundle(X=matrix, y=labels, feature_names=feature_names)


def _min_budget(rows: list[dict], *, family: str, threshold: float) -> dict:
    candidates = [
        row
        for row in rows
        if row["family"] == family and row["mean_recovery"] >= threshold
    ]
    if not candidates:
        return {
            "family": family,
            "threshold": threshold,
            "min_budget": None,
            "min_dim": None,
            "mean_recovery": None,
            "mean_accuracy": None,
        }
    best = min(candidates, key=lambda row: (row["dim"], row["budget"]))
    return {
        "family": family,
        "threshold": threshold,
        "min_budget": best["budget"],
        "min_dim": best["dim"],
        "mean_recovery": best["mean_recovery"],
        "mean_accuracy": best["mean_accuracy"],
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(output_dir: Path, aggregate_rows: list[dict], fixed_mean: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    styles = {
        "hashed_fixed_layout": ("#c44e52", "o", "Hashed fixed layout"),
        "coarse_count": ("#55a868", "D", "Coarse count"),
        "spectral_shape": ("#8172b2", "s", "Spectral shape"),
        "graphlet_orbit": ("#4c72b0", "^", "Graphlet orbit bins"),
        "graphlet_shape": ("#ccb974", "P", "Graphlet shape (canonical)"),
    }
    for family, (color, marker, label) in styles.items():
        rows = sorted(
            [row for row in aggregate_rows if row["family"] == family],
            key=lambda row: row["dim"],
        )
        if not rows:
            continue
        xs = [row["dim"] for row in rows]
        ys = [row["mean_recovery"] for row in rows]
        if len(rows) == 1:
            ax.scatter(
                xs,
                ys,
                marker=marker,
                color=color,
                edgecolor="#222222",
                linewidth=0.7,
                s=72,
                label=label,
                zorder=4,
            )
            ax.annotate(
                "fixed",
                xy=(xs[0], ys[0]),
                xytext=(7, 5),
                textcoords="offset points",
                fontsize=7.5,
                color=color,
            )
        else:
            ax.plot(
                xs,
                ys,
                marker=marker,
                color=color,
                linewidth=2,
                markersize=5,
                label=label,
            )

    ax.axhline(0.95, color="#333333", linestyle="--", linewidth=1.1, label="95% of fixed layout")
    ax.axhline(0.90, color="#777777", linestyle=":", linewidth=1.0, label="90% of fixed layout")
    ax.set_xscale("log")
    ax.set_ylim(0.45, 1.04)
    ax.set_xlabel("Feature dimension (log scale)")
    ax.set_ylabel("Accuracy / fixed-layout signed accuracy")
    ax.set_title("Feature-Budget Sweeps and Fixed-Budget References")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(fontsize=8, loc="lower right")
    ax.text(
        0.015,
        0.035,
        f"Fixed-layout signed mean accuracy = {fixed_mean:.4f}",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#333333",
    )
    fig.tight_layout()
    png_path = output_dir / "feature_budget_recovery.png"
    pdf_path = output_dir / "feature_budget_recovery.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def run(args: argparse.Namespace) -> None:
    seeds = _parse_ints(args.seeds)
    hash_dims = _parse_ints(args.hash_dims)
    spectral_values = _parse_ints(args.spectral_values)
    coarse_bin_pairs = _parse_bin_pairs(args.coarse_bins)
    graphlet_bin_pairs = _parse_bin_pairs(args.graphlet_bins)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict] = []
    fixed_accs: list[float] = []

    builder = create_graph_builder(args.graph_builder, k=args.k, enforce_direction=True)

    for seed in seeds:
        print(f"[seed {seed}] loading cached tensors", flush=True)
        run_key = f"seed{seed}_k{args.k}_nodes-res_n{args.num_examples}_paper_decision"
        cache_dir = Path(args.cache_root) / run_key
        dataset = _load_dataset(cache_dir)
        train_dataset, test_dataset, split_meta = _split_dataset_by_slice(
            dataset,
            train_fraction=args.train_fraction,
            seed=seed,
        )
        max_tokens = min(tensor.num_tokens for tensor in dataset)
        train_graphs = _build_bootstrap_graphs(
            train_dataset,
            builder,
            max_tokens=max_tokens,
            graphs_per_slice=args.graphs_per_slice,
            sample_fraction=args.bootstrap_sample_fraction,
            min_examples=args.bootstrap_min_examples,
            seed=seed + 10_000,
        )
        test_graphs = _build_bootstrap_graphs(
            test_dataset,
            builder,
            max_tokens=max_tokens,
            graphs_per_slice=args.graphs_per_slice,
            sample_fraction=args.bootstrap_sample_fraction,
            min_examples=args.bootstrap_min_examples,
            seed=seed + 20_000,
        )

        fixed_train = _fixed_layout_matrix(train_graphs, mode="sign_topology")
        fixed_test = _fixed_layout_matrix(
            test_graphs,
            feature_names=fixed_train.feature_names,
            mode="sign_topology",
        )
        fixed_acc = _eval_pair(fixed_train, fixed_test, seed=seed)
        fixed_accs.append(fixed_acc)
        per_seed_rows.append(
            {
                "seed": seed,
                "family": "fixed_layout_signed",
                "budget": len(fixed_train.feature_names),
                "dim": len(fixed_train.feature_names),
                "accuracy": fixed_acc,
                "fixed_accuracy": fixed_acc,
                "recovery": 1.0,
                "split": json.dumps(split_meta, sort_keys=True),
            }
        )

        for dim in hash_dims:
            print(f"[seed {seed}] hashed dim={dim}", flush=True)
            train = _hashed_fixed_layout_matrix(train_graphs, dim=dim, mode="sign")
            test = _hashed_fixed_layout_matrix(test_graphs, dim=dim, mode="sign")
            acc = _eval_pair(train, test, seed=seed)
            per_seed_rows.append(
                {
                    "seed": seed,
                    "family": "hashed_fixed_layout",
                    "budget": dim,
                    "dim": train.X.shape[1],
                    "accuracy": acc,
                    "fixed_accuracy": fixed_acc,
                    "recovery": acc / fixed_acc if fixed_acc else float("nan"),
                    "split": json.dumps(split_meta, sort_keys=True),
                }
            )

        for layer_bins, token_bins in coarse_bin_pairs:
            print(f"[seed {seed}] coarse bins={layer_bins}x{token_bins}", flush=True)
            coarse_train = _coarse_budget_matrix(
                train_graphs,
                layer_bins=layer_bins,
                token_bins=token_bins,
            )
            coarse_test = _coarse_budget_matrix(
                test_graphs,
                layer_bins=layer_bins,
                token_bins=token_bins,
                feature_names=coarse_train.feature_names,
            )
            acc = _eval_pair(coarse_train, coarse_test, seed=seed)
            per_seed_rows.append(
                {
                    "seed": seed,
                    "family": "coarse_count",
                    "budget": layer_bins * token_bins,
                    "dim": coarse_train.X.shape[1],
                    "accuracy": acc,
                    "fixed_accuracy": fixed_acc,
                    "recovery": acc / fixed_acc if fixed_acc else float("nan"),
                    "split": json.dumps(split_meta, sort_keys=True),
                }
            )

        for values in spectral_values:
            print(f"[seed {seed}] spectral values={values}", flush=True)
            train = _spectral_matrix(train_graphs, num_values=values)
            test = _spectral_matrix(test_graphs, num_values=values)
            acc = _eval_pair(train, test, seed=seed)
            per_seed_rows.append(
                {
                    "seed": seed,
                    "family": "spectral_shape",
                    "budget": values,
                    "dim": train.X.shape[1],
                    "accuracy": acc,
                    "fixed_accuracy": fixed_acc,
                    "recovery": acc / fixed_acc if fixed_acc else float("nan"),
                    "split": json.dumps(split_meta, sort_keys=True),
                }
            )

        for layer_bins, token_bins in graphlet_bin_pairs:
            print(f"[seed {seed}] graphlet orbit bins={layer_bins}x{token_bins}", flush=True)
            train = _graphlet_orbit_matrix(
                train_graphs,
                layer_bins=layer_bins,
                token_bins=token_bins,
            )
            test = _graphlet_orbit_matrix(
                test_graphs,
                layer_bins=layer_bins,
                token_bins=token_bins,
            )
            acc = _eval_pair(train, test, seed=seed)
            per_seed_rows.append(
                {
                    "seed": seed,
                    "family": "graphlet_orbit",
                    "budget": layer_bins * token_bins,
                    "dim": train.X.shape[1],
                    "accuracy": acc,
                    "fixed_accuracy": fixed_acc,
                    "recovery": acc / fixed_acc if fixed_acc else float("nan"),
                    "split": json.dumps(split_meta, sort_keys=True),
                }
            )

        train = _graphlet_matrix(train_graphs)
        test = _graphlet_matrix(test_graphs)
        acc = _eval_pair(train, test, seed=seed)
        per_seed_rows.append(
            {
                "seed": seed,
                "family": "graphlet_shape",
                "budget": train.X.shape[1],
                "dim": train.X.shape[1],
                "accuracy": acc,
                "fixed_accuracy": fixed_acc,
                "recovery": acc / fixed_acc if fixed_acc else float("nan"),
                "split": json.dumps(split_meta, sort_keys=True),
            }
        )

    fieldnames = [
        "seed",
        "family",
        "budget",
        "dim",
        "accuracy",
        "fixed_accuracy",
        "recovery",
        "split",
    ]
    _write_csv(output_dir / "feature_budget_per_seed.csv", per_seed_rows, fieldnames)

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in per_seed_rows:
        grouped[(row["family"], int(row["budget"]))].append(row)

    aggregate_rows = []
    for (family, budget), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        accs = [float(row["accuracy"]) for row in rows]
        recoveries = [float(row["recovery"]) for row in rows]
        dims = [int(row["dim"]) for row in rows]
        aggregate_rows.append(
            {
                "family": family,
                "budget": budget,
                "dim": mean(dims),
                "dim_min": min(dims),
                "dim_max": max(dims),
                "mean_accuracy": mean(accs),
                "std_accuracy": pstdev(accs) if len(accs) > 1 else 0.0,
                "mean_recovery": mean(recoveries),
                "std_recovery": pstdev(recoveries) if len(recoveries) > 1 else 0.0,
                "seeds": len(rows),
            }
        )

    agg_fields = [
        "family",
        "budget",
        "dim",
        "dim_min",
        "dim_max",
        "mean_accuracy",
        "std_accuracy",
        "mean_recovery",
        "std_recovery",
        "seeds",
    ]
    _write_csv(output_dir / "feature_budget_summary.csv", aggregate_rows, agg_fields)

    threshold_rows = []
    for threshold in (0.90, 0.95):
        for family in (
            "hashed_fixed_layout",
            "coarse_count",
            "spectral_shape",
            "graphlet_orbit",
            "graphlet_shape",
        ):
            threshold_rows.append(_min_budget(aggregate_rows, family=family, threshold=threshold))
    _write_csv(
        output_dir / "feature_budget_thresholds.csv",
        threshold_rows,
        ["family", "threshold", "min_budget", "min_dim", "mean_recovery", "mean_accuracy"],
    )

    fixed_mean = mean(fixed_accs)
    fig_path = _plot_results(output_dir, aggregate_rows, fixed_mean)
    metadata = {
        "cache_root": args.cache_root,
        "seeds": seeds,
        "num_examples": args.num_examples,
        "k": args.k,
        "fixed_layout_signed_mean_accuracy": fixed_mean,
        "hash_dims": hash_dims,
        "spectral_values": spectral_values,
        "coarse_bins": coarse_bin_pairs,
        "graphlet_bins": graphlet_bin_pairs,
        "figure": str(fig_path),
    }
    (output_dir / "feature_budget_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(output_dir / "feature_budget_summary.csv")
    print(output_dir / "feature_budget_thresholds.csv")
    print(fig_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default=".cache/paper_decision/gpt2_n100_res_k5_20260430")
    parser.add_argument("--output-dir", default="outputs/feature_budget_sweep/gpt2_n100_res_k5")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--graph-builder", default="correlation_topk")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--graphs-per-slice", type=int, default=32)
    parser.add_argument("--bootstrap-sample-fraction", type=float, default=0.75)
    parser.add_argument("--bootstrap-min-examples", type=int, default=3)
    parser.add_argument("--hash-dims", default="32,64,128,256,512,1024,2048,4096")
    parser.add_argument("--spectral-values", default="4,8,16,32")
    parser.add_argument("--coarse-bins", default="1x1,2x2,4x4,8x8")
    parser.add_argument("--graphlet-bins", default="1x1,2x2,4x4,6x6")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
