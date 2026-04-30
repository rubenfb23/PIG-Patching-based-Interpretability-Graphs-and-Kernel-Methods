#!/usr/bin/env python3
"""Run stricter representation checks for paper decisions.

This runner avoids the most optimistic bootstrap-CV setting by splitting base
examples before graph bootstrapping. Train graphs and test graphs are therefore
built from disjoint prompt pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from pig.embeddings import WLEncoder
from pig.graph import (
    GraphBuilder,
    Node,
    PatchInfluenceGraph,
    create_graph_builder,
)
from pig.graph_features import apply_null_control_to_graphs
from pig.model import create_model
from pig.patching import PatchEffectDataset, compute_patch_effects
from pig.prompts import SliceLabel, create_ioi_dataset


@dataclass
class MatrixBundle:
    X: np.ndarray
    y: list[SliceLabel]
    feature_names: list[str]


def _parse_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_grid(raw: str) -> list[int]:
    return [int(part) for part in _parse_csv_values(raw)]


def _parse_node_type_grid(raw: str) -> list[tuple[str, ...]]:
    groups = []
    for part in raw.split(";"):
        values = tuple(_parse_csv_values(part))
        if values:
            groups.append(values)
    return groups


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_key(
    *,
    seed: int,
    k: int,
    node_types: Sequence[str],
    num_examples: int,
) -> str:
    node_key = "-".join(node_types)
    return f"seed{seed}_k{k}_nodes-{node_key}_n{num_examples}_paper_decision"


def _subset_dataset(tensors: Iterable) -> PatchEffectDataset:
    dataset = PatchEffectDataset()
    for tensor in tensors:
        dataset.add(tensor)
    return dataset


def _split_dataset_by_slice(
    dataset: PatchEffectDataset,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[PatchEffectDataset, PatchEffectDataset, dict]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_tensors = []
    test_tensors = []
    split_meta = {}

    for slice_label in sorted(dataset.get_slices(), key=str):
        tensors = dataset.get_by_slice(slice_label)
        if len(tensors) < 4:
            raise ValueError(f"Need at least four tensors for slice {slice_label}")
        order = rng.permutation(len(tensors))
        train_size = int(math.floor(len(tensors) * train_fraction))
        train_size = min(max(train_size, 2), len(tensors) - 2)
        train_indices = order[:train_size]
        test_indices = order[train_size:]

        train_tensors.extend(tensors[int(index)] for index in train_indices)
        test_tensors.extend(tensors[int(index)] for index in test_indices)
        split_meta[str(slice_label)] = {
            "total": len(tensors),
            "train": len(train_indices),
            "test": len(test_indices),
        }

    return _subset_dataset(train_tensors), _subset_dataset(test_tensors), split_meta


def _build_bootstrap_graphs(
    dataset: PatchEffectDataset,
    builder: GraphBuilder,
    *,
    max_tokens: int,
    graphs_per_slice: int,
    sample_fraction: float,
    min_examples: int,
    seed: int,
    replace: bool = True,
) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
    rng = np.random.default_rng(seed)
    graphs: list[tuple[PatchInfluenceGraph, SliceLabel]] = []

    for slice_label in sorted(dataset.get_slices(), key=str):
        tensors = dataset.get_by_slice(slice_label)
        if not tensors:
            continue

        requested_size = max(
            min_examples,
            int(math.ceil(len(tensors) * sample_fraction)),
        )
        sample_size = requested_size if replace else min(requested_size, len(tensors))

        for bootstrap_index in range(graphs_per_slice):
            indices = rng.choice(
                len(tensors),
                size=sample_size,
                replace=replace or sample_size > len(tensors),
            )
            sampled_tensors = [tensors[int(index)] for index in indices]
            graph = builder.build_from_tensors(
                sampled_tensors,
                slice_label,
                max_tokens=max_tokens,
                graph_role="paper_decision_bootstrap",
                construction_mode="example_disjoint_bootstrap_slice_correlation",
                metadata={
                    "bootstrap_index": bootstrap_index,
                    "bootstrap_sample_size": sample_size,
                    "bootstrap_sample_fraction": float(sample_fraction),
                    "bootstrap_replace": bool(replace),
                },
            )
            graphs.append((graph, slice_label))

    return graphs


def _node_key(node: Node) -> str:
    head = "none" if node.head is None else str(node.head)
    return f"L{node.layer}:T{node.token}:{node.node_type}:H{head}"


def _edge_key(src: Node, dst: Node) -> str:
    return f"edge|{_node_key(src)}->{_node_key(dst)}"


def _fixed_layout_vocabulary(graphs: list[PatchInfluenceGraph]) -> list[str]:
    if not graphs:
        return []
    nodes = graphs[0].nodes
    features = [_edge_key(src, dst) for src in nodes for dst in nodes if src != dst]
    return sorted(features)


def _fixed_layout_matrix(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    *,
    feature_names: list[str] | None = None,
    mode: str = "weighted",
) -> MatrixBundle:
    graphs = [graph for graph, _ in graphs_with_labels]
    labels = [label for _, label in graphs_with_labels]
    if feature_names is None:
        feature_names = _fixed_layout_vocabulary(graphs)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    matrix = np.zeros((len(graphs), len(feature_names)), dtype=np.float32)

    for row_idx, graph in enumerate(graphs):
        for edge in graph.edges:
            src = graph.nodes[edge.src]
            dst = graph.nodes[edge.dst]
            col_idx = feature_index.get(_edge_key(src, dst))
            if col_idx is None:
                continue
            if mode == "weighted":
                value = edge.weight
            elif mode == "binary_topology":
                value = 1.0
            elif mode == "sign_topology":
                value = np.sign(edge.weight)
            else:
                raise ValueError(f"Unknown fixed-layout mode: {mode}")
            matrix[row_idx, col_idx] += np.float32(value)

    return MatrixBundle(X=matrix, y=labels, feature_names=feature_names)


def _wl_matrix(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    *,
    depth: int,
    vocabulary: list[str] | None = None,
) -> MatrixBundle:
    graphs = [graph for graph, _ in graphs_with_labels]
    labels = [label for _, label in graphs_with_labels]
    encoder = WLEncoder(depth=depth)
    embeddings = encoder.encode_batch(graphs)
    if vocabulary is None:
        feature_set = set()
        for embedding in embeddings:
            feature_set.update(embedding.features.keys())
        vocabulary = sorted(feature_set)
    matrix = np.stack([embedding.to_vector(vocabulary) for embedding in embeddings])
    return MatrixBundle(X=matrix, y=labels, feature_names=vocabulary)


def _dense_adjacency(
    graph: PatchInfluenceGraph,
    *,
    mode: str,
    symmetric: bool = True,
) -> np.ndarray:
    adjacency = np.zeros((graph.num_nodes, graph.num_nodes), dtype=np.float64)
    for edge in graph.edges:
        if mode == "weighted":
            value = edge.weight
        elif mode == "absolute":
            value = abs(edge.weight)
        elif mode == "binary":
            value = 1.0
        elif mode == "sign":
            value = float(np.sign(edge.weight))
        else:
            raise ValueError(f"Unknown adjacency mode: {mode}")
        adjacency[edge.src, edge.dst] += value

    if symmetric:
        adjacency = adjacency + adjacency.T
    return adjacency


def _sparse_adjacency(
    graph: PatchInfluenceGraph,
    *,
    mode: str,
    symmetric: bool = True,
) -> csr_matrix:
    rows = []
    cols = []
    values = []
    for edge in graph.edges:
        if mode == "weighted":
            value = edge.weight
        elif mode == "absolute":
            value = abs(edge.weight)
        elif mode == "binary":
            value = 1.0
        elif mode == "sign":
            value = float(np.sign(edge.weight))
        else:
            raise ValueError(f"Unknown adjacency mode: {mode}")
        rows.append(edge.src)
        cols.append(edge.dst)
        values.append(value)

    adjacency = csr_matrix(
        (values, (rows, cols)),
        shape=(graph.num_nodes, graph.num_nodes),
        dtype=np.float64,
    )
    if symmetric:
        adjacency = adjacency + adjacency.T
    return adjacency


def _pad_vector(values: np.ndarray, size: int) -> np.ndarray:
    padded = np.zeros(size, dtype=np.float64)
    count = min(size, values.size)
    if count:
        padded[:count] = values[:count]
    return padded


def _extreme_eigenpairs(
    matrix: csr_matrix,
    *,
    num_values: int,
    which: str,
    with_vectors: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    n = matrix.shape[0]
    k = min(num_values, n)
    if k == 0:
        empty_vectors = np.zeros((n, 0), dtype=np.float64) if with_vectors else None
        return np.zeros(0, dtype=np.float64), empty_vectors

    # ARPACK requires k < n. For tiny graphs, dense fallback is simpler and exact.
    if k >= n:
        dense = matrix.toarray()
        values, vectors = np.linalg.eigh(dense)
    else:
        try:
            values, vectors = eigsh(
                matrix,
                k=k,
                which=which,
                return_eigenvectors=True,
                tol=1e-4,
                maxiter=max(1000, 20 * n),
            )
        except ArpackNoConvergence as exc:
            if exc.eigenvalues.size >= k:
                values = exc.eigenvalues
                vectors = exc.eigenvectors
            else:
                dense = matrix.toarray()
                values, vectors = np.linalg.eigh(dense)

    order = np.argsort(values)
    if which in {"LA", "LM"}:
        order = order[-k:]
    else:
        order = order[:k]
    values = values[order]
    vectors = vectors[:, order]
    return values, vectors if with_vectors else None


def _spectral_features(
    graph: PatchInfluenceGraph,
    *,
    num_values: int,
) -> np.ndarray:
    """Compute compact spectral shape features from graph matrices."""
    abs_adjacency = _sparse_adjacency(graph, mode="absolute", symmetric=True)
    signed_adjacency = _sparse_adjacency(graph, mode="sign", symmetric=True)

    degree = np.asarray(abs_adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 1e-12
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    scale = diags(inv_sqrt)
    normalized = scale @ abs_adjacency @ scale
    laplacian = eye(graph.num_nodes, dtype=np.float64, format="csr") - normalized

    lap_low, low_vectors = _extreme_eigenpairs(
        laplacian,
        num_values=num_values,
        which="SA",
        with_vectors=True,
    )
    lap_high, high_vectors = _extreme_eigenpairs(
        laplacian,
        num_values=num_values,
        which="LA",
        with_vectors=True,
    )
    signed_low, _ = _extreme_eigenpairs(
        signed_adjacency,
        num_values=num_values,
        which="SA",
        with_vectors=False,
    )
    signed_high, _ = _extreme_eigenpairs(
        signed_adjacency,
        num_values=num_values,
        which="LA",
        with_vectors=False,
    )

    low_ipr = np.sum(low_vectors**4, axis=0) if low_vectors is not None else lap_low
    high_ipr = np.sum(high_vectors**4, axis=0) if high_vectors is not None else lap_high

    return np.concatenate(
        [
            _pad_vector(lap_low, num_values),
            _pad_vector(lap_high, num_values),
            _pad_vector(signed_low, num_values),
            _pad_vector(signed_high, num_values),
            _pad_vector(low_ipr, num_values),
            _pad_vector(high_ipr, num_values),
        ],
    ).astype(np.float32)


def _spectral_matrix(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    *,
    num_values: int,
) -> MatrixBundle:
    matrix = np.stack(
        [
            _spectral_features(graph, num_values=num_values)
            for graph, _ in graphs_with_labels
        ]
    )
    labels = [label for _, label in graphs_with_labels]
    actual_values = matrix.shape[1] // 6
    feature_names = (
        [f"lap_low_{idx}" for idx in range(actual_values)]
        + [f"lap_high_{idx}" for idx in range(actual_values)]
        + [f"signed_adj_low_{idx}" for idx in range(actual_values)]
        + [f"signed_adj_high_{idx}" for idx in range(actual_values)]
        + [f"lap_low_ipr_{idx}" for idx in range(actual_values)]
        + [f"lap_high_ipr_{idx}" for idx in range(actual_values)]
    )
    return MatrixBundle(X=matrix, y=labels, feature_names=feature_names)


def _safe_stats(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    mean = float(np.mean(values))
    std = float(np.std(values))
    max_value = float(np.max(values))
    if std <= 1e-12:
        skew = 0.0
    else:
        skew = float(np.mean(((values - mean) / std) ** 3))
    return [mean, std, max_value, skew]


def _graphlet_features(graph: PatchInfluenceGraph) -> np.ndarray:
    """Compute low-dimensional 3-node graphlet and degree statistics."""
    binary = _dense_adjacency(graph, mode="binary", symmetric=True)
    binary = (binary > 0).astype(np.float64)
    n = graph.num_nodes
    m = int(np.sum(binary) / 2)

    total_triplets = n * (n - 1) * (n - 2) / 6 if n >= 3 else 0.0
    degree = binary.sum(axis=1)
    wedge_center_count = float(np.sum(degree * (degree - 1) / 2))
    triangles = float(np.trace(binary @ binary @ binary) / 6.0)
    two_edge = max(wedge_center_count - 3.0 * triangles, 0.0)
    one_edge = max(m * (n - 2) - 2.0 * two_edge - 3.0 * triangles, 0.0)
    zero_edge = max(total_triplets - one_edge - two_edge - triangles, 0.0)
    triplet_den = max(total_triplets, 1.0)
    wedge_den = max(wedge_center_count, 1.0)

    weights = np.array([edge.weight for edge in graph.edges], dtype=np.float64)
    pos_weights = weights[weights > 0]
    neg_weights = weights[weights < 0]
    abs_weights = np.abs(weights)

    signed = _dense_adjacency(graph, mode="sign", symmetric=True)
    pos_degree = (signed > 0).sum(axis=1).astype(np.float64)
    neg_degree = (signed < 0).sum(axis=1).astype(np.float64)

    features = [
        float(n),
        float(m),
        float(m / max(n * (n - 1) / 2, 1.0)),
        zero_edge / triplet_den,
        one_edge / triplet_den,
        two_edge / triplet_den,
        triangles / triplet_den,
        triangles / wedge_den,
        float(len(pos_weights) / max(len(weights), 1)),
        float(len(neg_weights) / max(len(weights), 1)),
        *_safe_stats(degree),
        *_safe_stats(pos_degree),
        *_safe_stats(neg_degree),
        *_safe_stats(weights),
        *_safe_stats(abs_weights),
        *_safe_stats(pos_weights),
        *_safe_stats(np.abs(neg_weights)),
    ]
    return np.array(features, dtype=np.float32)


def _graphlet_matrix(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
) -> MatrixBundle:
    matrix = np.stack([_graphlet_features(graph) for graph, _ in graphs_with_labels])
    labels = [label for _, label in graphs_with_labels]
    feature_names = [
        "num_nodes",
        "num_edges",
        "density",
        "triplets_0_edges",
        "triplets_1_edge",
        "triplets_2_edges",
        "triangles",
        "transitivity",
        "positive_edge_fraction",
        "negative_edge_fraction",
    ]
    for prefix in (
        "degree",
        "positive_degree",
        "negative_degree",
        "weight",
        "abs_weight",
        "positive_weight",
        "negative_abs_weight",
    ):
        feature_names.extend(
            [
                f"{prefix}_mean",
                f"{prefix}_std",
                f"{prefix}_max",
                f"{prefix}_skew",
            ]
        )
    return MatrixBundle(X=matrix, y=labels, feature_names=feature_names)


def _fit_eval_svm(
    train: MatrixBundle,
    test: MatrixBundle,
    *,
    kernel: str,
    seed: int,
) -> dict:
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform([str(label) for label in train.y])
    y_test = label_encoder.transform([str(label) for label in test.y])
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=kernel,
                    C=1.0,
                    gamma="scale",
                    random_state=seed,
                    probability=False,
                ),
            ),
        ]
    )
    estimator.fit(train.X, y_train)
    predictions = estimator.predict(test.X)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "num_train": int(train.X.shape[0]),
        "num_test": int(test.X.shape[0]),
        "num_features": int(train.X.shape[1]),
        "kernel": kernel,
    }


def _null_test_accuracy(
    train_bundle: MatrixBundle,
    test_graphs: list[tuple[PatchInfluenceGraph, SliceLabel]],
    *,
    mode: str,
    kernel: str,
    control: str,
    repeats: int,
    seed: int,
) -> dict:
    values = []
    for repeat in range(repeats):
        null_graphs = apply_null_control_to_graphs(
            test_graphs,
            control,
            seed=seed + repeat,
        )
        null_bundle = _fixed_layout_matrix(
            null_graphs,
            feature_names=train_bundle.feature_names,
            mode=mode,
        )
        values.append(
            _fit_eval_svm(
                train_bundle,
                null_bundle,
                kernel=kernel,
                seed=seed + repeat,
            )["accuracy"]
        )
    return {
        "accuracy_mean": float(np.mean(values)) if values else float("nan"),
        "accuracy_std": float(np.std(values)) if values else float("nan"),
        "values": values,
        "repeats": int(repeats),
    }


def _evaluate_seed(
    *,
    dataset: PatchEffectDataset,
    builder: GraphBuilder,
    seed: int,
    max_tokens: int,
    train_fraction: float,
    graphs_per_slice: int,
    sample_fraction: float,
    min_examples: int,
    wl_depth: int,
    spectral_values: int,
    null_repeats: int,
) -> dict:
    train_dataset, test_dataset, split_meta = _split_dataset_by_slice(
        dataset,
        train_fraction=train_fraction,
        seed=seed,
    )
    train_graphs = _build_bootstrap_graphs(
        train_dataset,
        builder,
        max_tokens=max_tokens,
        graphs_per_slice=graphs_per_slice,
        sample_fraction=sample_fraction,
        min_examples=min_examples,
        seed=seed + 10_000,
    )
    test_graphs = _build_bootstrap_graphs(
        test_dataset,
        builder,
        max_tokens=max_tokens,
        graphs_per_slice=graphs_per_slice,
        sample_fraction=sample_fraction,
        min_examples=min_examples,
        seed=seed + 20_000,
    )

    result = {
        "split": split_meta,
        "num_train_graphs": len(train_graphs),
        "num_test_graphs": len(test_graphs),
        "metrics": {},
    }

    wl_train = _wl_matrix(train_graphs, depth=wl_depth)
    wl_test = _wl_matrix(test_graphs, depth=wl_depth, vocabulary=wl_train.feature_names)
    result["metrics"]["wl_bootstrap"] = {
        kernel: _fit_eval_svm(wl_train, wl_test, kernel=kernel, seed=seed)
        for kernel in ("linear", "rbf")
    }

    spectral_train = _spectral_matrix(train_graphs, num_values=spectral_values)
    spectral_test = _spectral_matrix(test_graphs, num_values=spectral_values)
    result["metrics"]["spectral_shape"] = {
        kernel: _fit_eval_svm(spectral_train, spectral_test, kernel=kernel, seed=seed)
        for kernel in ("linear", "rbf")
    }

    graphlet_train = _graphlet_matrix(train_graphs)
    graphlet_test = _graphlet_matrix(test_graphs)
    result["metrics"]["graphlet_shape"] = {
        kernel: _fit_eval_svm(graphlet_train, graphlet_test, kernel=kernel, seed=seed)
        for kernel in ("linear", "rbf")
    }

    for mode in ("weighted", "binary_topology", "sign_topology"):
        train_bundle = _fixed_layout_matrix(train_graphs, mode=mode)
        test_bundle = _fixed_layout_matrix(
            test_graphs,
            feature_names=train_bundle.feature_names,
            mode=mode,
        )
        family = {
            kernel: _fit_eval_svm(
                train_bundle,
                test_bundle,
                kernel=kernel,
                seed=seed,
            )
            for kernel in ("linear", "rbf")
        }
        if mode == "weighted":
            family["null_test_controls"] = {
                control: _null_test_accuracy(
                    train_bundle,
                    test_graphs,
                    mode=mode,
                    kernel="linear",
                    control=control,
                    repeats=null_repeats,
                    seed=seed + 30_000,
                )
                for control in ("edge_shuffle", "weight_shuffle")
            }
        result["metrics"][f"fixed_layout_{mode}"] = family

    return result


def _write_summary(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run_key",
        "seed",
        "wl_linear",
        "wl_rbf",
        "fixed_weighted_linear",
        "fixed_weighted_rbf",
        "fixed_binary_linear",
        "fixed_binary_rbf",
        "fixed_sign_linear",
        "fixed_sign_rbf",
        "spectral_linear",
        "spectral_rbf",
        "graphlet_linear",
        "graphlet_rbf",
        "fixed_weighted_edge_shuffle_linear",
        "fixed_weighted_weight_shuffle_linear",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_paper_decision_study",
        description="Run example-disjoint representation checks for paper decisions.",
    )
    parser.add_argument("--model-name", default="toy_transformer")
    parser.add_argument("--device", default=None)
    parser.add_argument("--corruptions", default="name_swap,abba")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--k-grid", default="5")
    parser.add_argument("--num-examples-grid", default="100")
    parser.add_argument("--node-types-grid", default="res")
    parser.add_argument("--graph-builder", default="correlation_topk")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--graphs-per-slice", type=int, default=32)
    parser.add_argument("--bootstrap-sample-fraction", type=float, default=0.75)
    parser.add_argument("--bootstrap-min-examples", type=int, default=3)
    parser.add_argument("--wl-depth", type=int, default=3)
    parser.add_argument("--spectral-values", type=int, default=16)
    parser.add_argument("--null-repeats", type=int, default=10)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = _parse_int_grid(args.seeds)
    k_grid = _parse_int_grid(args.k_grid)
    num_examples_grid = _parse_int_grid(args.num_examples_grid)
    node_type_grid = _parse_node_type_grid(args.node_types_grid)
    corruptions = _parse_csv_values(args.corruptions)
    if not corruptions:
        raise ValueError("At least one corruption must be provided")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs")
        / "paper_decision"
        / f"{args.model_name.replace('/', '_')}_{_utc_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(args.model_name, device=args.device)
    summary_rows = []

    for seed in seeds:
        for k in k_grid:
            for node_types in node_type_grid:
                for num_examples in num_examples_grid:
                    run_key = _run_key(
                        seed=seed,
                        k=k,
                        node_types=node_types,
                        num_examples=num_examples,
                    )
                    print(f"Running {run_key}")
                    prompt_pairs = []
                    for offset, corruption in enumerate(corruptions):
                        prompt_pairs.extend(
                            create_ioi_dataset(
                                n_examples=num_examples,
                                corruption=corruption,
                                seed=seed + offset * 10_000,
                            )
                        )

                    cache_dir = None
                    if args.cache_root:
                        cache_dir = str(Path(args.cache_root) / run_key)

                    dataset = compute_patch_effects(
                        model,
                        prompt_pairs,
                        cache_dir=cache_dir,
                        show_progress=True,
                        node_types=node_types,
                    )
                    max_tokens = min(tensor.num_tokens for tensor in dataset)
                    builder = create_graph_builder(
                        args.graph_builder,
                        k=k,
                        enforce_direction=True,
                    )
                    result = _evaluate_seed(
                        dataset=dataset,
                        builder=builder,
                        seed=seed,
                        max_tokens=max_tokens,
                        train_fraction=args.train_fraction,
                        graphs_per_slice=args.graphs_per_slice,
                        sample_fraction=args.bootstrap_sample_fraction,
                        min_examples=args.bootstrap_min_examples,
                        wl_depth=args.wl_depth,
                        spectral_values=args.spectral_values,
                        null_repeats=args.null_repeats,
                    )
                    report = {
                        "run_key": run_key,
                        "model_name": args.model_name,
                        "seed": seed,
                        "k": k,
                        "node_types": list(node_types),
                        "num_examples_per_corruption": num_examples,
                        "corruptions": corruptions,
                        "max_tokens": max_tokens,
                        "paper_decision": result,
                    }
                    with (runs_dir / f"{run_key}.json").open(
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        json.dump(report, handle, indent=2, sort_keys=True)

                    metrics = result["metrics"]
                    weighted_nulls = metrics["fixed_layout_weighted"][
                        "null_test_controls"
                    ]
                    summary_rows.append(
                        {
                            "run_key": run_key,
                            "seed": seed,
                            "wl_linear": metrics["wl_bootstrap"]["linear"]["accuracy"],
                            "wl_rbf": metrics["wl_bootstrap"]["rbf"]["accuracy"],
                            "fixed_weighted_linear": metrics["fixed_layout_weighted"][
                                "linear"
                            ]["accuracy"],
                            "fixed_weighted_rbf": metrics["fixed_layout_weighted"][
                                "rbf"
                            ]["accuracy"],
                            "fixed_binary_linear": metrics[
                                "fixed_layout_binary_topology"
                            ]["linear"]["accuracy"],
                            "fixed_binary_rbf": metrics["fixed_layout_binary_topology"][
                                "rbf"
                            ]["accuracy"],
                            "fixed_sign_linear": metrics["fixed_layout_sign_topology"][
                                "linear"
                            ]["accuracy"],
                            "fixed_sign_rbf": metrics["fixed_layout_sign_topology"][
                                "rbf"
                            ]["accuracy"],
                            "spectral_linear": metrics["spectral_shape"]["linear"][
                                "accuracy"
                            ],
                            "spectral_rbf": metrics["spectral_shape"]["rbf"][
                                "accuracy"
                            ],
                            "graphlet_linear": metrics["graphlet_shape"]["linear"][
                                "accuracy"
                            ],
                            "graphlet_rbf": metrics["graphlet_shape"]["rbf"][
                                "accuracy"
                            ],
                            "fixed_weighted_edge_shuffle_linear": weighted_nulls[
                                "edge_shuffle"
                            ]["accuracy_mean"],
                            "fixed_weighted_weight_shuffle_linear": weighted_nulls[
                                "weight_shuffle"
                            ]["accuracy_mean"],
                        }
                    )
                    _write_summary(output_dir / "summary.csv", summary_rows)

    print(f"Wrote {len(summary_rows)} paper-decision reports to {output_dir}")


if __name__ == "__main__":
    main()
