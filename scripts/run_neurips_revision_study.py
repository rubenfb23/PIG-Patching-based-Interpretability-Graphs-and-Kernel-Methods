#!/usr/bin/env python3
"""Run NeurIPS revision experiments for PIG.

Usage:
    # Full sweep
    python scripts/run_neurips_revision_study.py \
        --tasks ioi greater_than kv_retrieval \
        --seeds 7,42,123,256,512,999,1337,2024,3141,9999 \
        --n-grid 20,50,100,200,500

    # Quick smoke test
    python scripts/run_neurips_revision_study.py \
        --tasks ioi --seeds 7,42,123 --n-grid 20,50,100

    # Hyperparameter ablations (only at n=100)
    python scripts/run_neurips_revision_study.py \
        --tasks ioi --seeds 7,42,123 --n-grid 100 \
        --k-ablation-grid 1,2,5,10,20

    # Dimensionality control
    python scripts/run_neurips_revision_study.py \
        --tasks ioi --seeds 7,42,123 --n-grid 100 \
        --dim-control

    # Patch-effect raw sweep
    python scripts/run_neurips_revision_study.py \
        --tasks ioi --seeds 7,42,123 --n-grid 100 --patch-effect-raw
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Use a writable HF cache (avoid stale root-owned .locks dir)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub2"))

from pig.causal import (
    CausalEvalConfig,
    build_clean_better_filter_report,
    evaluate_causal_edges,
    propose_causal_edge_candidates,
    propose_random_edge_candidates,
    propose_lowrank_edge_candidates,
    split_discovery_evaluation_tensors,
)
from pig.embeddings import compute_wl_features_from_list
from pig.graph import build_bootstrap_slice_graphs, create_graph_builder
from pig.graph_features import (
    apply_null_control_to_graphs,
    compute_fixed_layout_features_from_list,
    compute_patch_effect_features_from_list,
    compute_topk_node_features_from_list,
    compute_histogram_features_from_list,
    compute_hashed_fixed_layout_features,
    compute_coarse_count_features,
    compute_label_permutation_features,
    compute_random_graph_features,
    compute_pca_projected_features,
    compute_random_projection_features,
)
from pig.kernels import ClassicalKernelClassifier
from pig.model import create_model
from pig.patching import compute_patch_effects
from pig.prompts import (
    create_ioi_dataset,
    create_greater_than_dataset,
    create_kv_dataset,
    validate_y_star,
)

# ---- Constants ----

TASK_GENERATORS = {
    "ioi": create_ioi_dataset,
    "greater_than": create_greater_than_dataset,
    "kv_retrieval": create_kv_dataset,
}


def _generate_task_pairs(task_name: str, n: int, seed: int, **kwargs) -> list:
    """Generate prompt pairs for a specific task."""
    if task_name == "ioi":
        return create_ioi_dataset(n, corruption=kwargs.get("corruption", "name_swap"), seed=seed)
    elif task_name == "greater_than":
        return create_greater_than_dataset(n, seed=seed)
    elif task_name == "kv_retrieval":
        return create_kv_dataset(n, n_pairs=kwargs.get("n_pairs", 10), seed=seed)
    else:
        raise ValueError(f"Unknown task: {task_name}")

DEFAULT_SEEDS = [7, 42, 123, 256, 512, 999, 1337, 2024, 3141, 9999]
DEFAULT_N_GRID = [20, 50, 100, 200, 500]


# ---- Helpers ----

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_grid(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _matrix_metrics(X: np.ndarray, y, seed: int) -> dict:
    metrics: dict[str, Any] = {}
    for kernel in ("linear", "rbf"):
        clf = ClassicalKernelClassifier(kernel=kernel, random_state=seed)
        cv = clf.cross_validate(X, y)
        metrics[kernel] = cv
    return metrics


def _aggregate_null(observed: dict, null_runs: list[dict]) -> dict:
    if not null_runs:
        return {}
    summary = {}
    for kernel in ("linear", "rbf"):
        values = [
            float(run[kernel]["accuracy_mean"])
            for run in null_runs
            if kernel in run and "accuracy_mean" in run[kernel]
        ]
        if not values:
            continue
        null_mean = float(np.mean(values))
        null_std = float(np.std(values))
        summary[kernel] = {
            "accuracy_mean": null_mean,
            "accuracy_std": null_std,
            "observed_minus_null_mean": float(
                observed[kernel]["accuracy_mean"] - null_mean
            ),
            "repeats": len(values),
            "accuracy_values": values,
        }
    return summary


def _evaluate_baseline(
    graphs_with_labels,
    *,
    feature_name: str,
    feature_builder,
    seed: int,
    null_repeats: int,
    graph_controls: list[str] | None = None,
) -> dict:
    """Evaluate one feature family with null controls."""
    if not graphs_with_labels:
        return {"error": "No graphs available", "feature_name": feature_name}

    feature_matrix = feature_builder(graphs_with_labels)
    X = feature_matrix.to_matrix()
    y = feature_matrix.slice_labels
    observed = _matrix_metrics(X, y, seed=seed)

    result = {
        "feature_name": feature_name,
        "feature_shape": {
            "num_graphs": int(feature_matrix.num_graphs),
            "num_features": int(feature_matrix.num_features),
        },
        "observed": observed,
    }

    if null_repeats <= 0:
        return result

    rng = np.random.default_rng(seed)

    # Label permutation null
    label_runs = []
    for repeat in range(null_repeats):
        order = rng.permutation(len(y))
        y_perm = [y[int(i)] for i in order]
        label_runs.append(_matrix_metrics(X, y_perm, seed=seed + 101 + repeat))
    result["null_controls"] = {"label_permutation": _aggregate_null(observed, label_runs)}

    # Edge shuffle null
    edge_runs = []
    for repeat in range(null_repeats):
        null_graphs = apply_null_control_to_graphs(
            graphs_with_labels, "edge_shuffle", seed=seed + 1_000 + repeat
        )
        fm = feature_builder(null_graphs)
        edge_runs.append(_matrix_metrics(fm.to_matrix(), fm.slice_labels, seed=seed + 2_000 + repeat))
    result["null_controls"]["edge_shuffle"] = _aggregate_null(observed, edge_runs)

    # Weight shuffle null
    weight_runs = []
    for repeat in range(null_repeats):
        null_graphs = apply_null_control_to_graphs(
            graphs_with_labels, "weight_shuffle", seed=seed + 3_000 + repeat
        )
        fm = feature_builder(null_graphs)
        weight_runs.append(_matrix_metrics(fm.to_matrix(), fm.slice_labels, seed=seed + 4_000 + repeat))
    result["null_controls"]["weight_shuffle"] = _aggregate_null(observed, weight_runs)

    # Additional graph-level controls (sign_shuffle, etc.)
    if graph_controls:
        for control in graph_controls:
            control_runs = []
            for repeat in range(null_repeats):
                null_graphs = apply_null_control_to_graphs(
                    graphs_with_labels, control, seed=seed + 5_000 + control.encode()[0] * 1000 + repeat
                )
                fm = feature_builder(null_graphs)
                control_runs.append(_matrix_metrics(fm.to_matrix(), fm.slice_labels, seed=seed + 6_000 + control.encode()[0] * 1000 + repeat))
            result["null_controls"][control] = _aggregate_null(observed, control_runs)

    return result


def _evaluate_baseline_raw(
    tensors,
    *,
    feature_name: str,
    feature_builder,
    seed: int,
    null_repeats: int,
) -> dict:
    """Evaluate a feature family that takes raw tensors (not graphs)."""
    if not tensors:
        return {"error": "No tensors available", "feature_name": feature_name}

    feature_matrix = feature_builder(tensors)
    X = feature_matrix.to_matrix()
    y = feature_matrix.slice_labels
    observed = _matrix_metrics(X, y, seed=seed)

    result = {
        "feature_name": feature_name,
        "feature_shape": {
            "num_graphs": int(feature_matrix.num_graphs),
            "num_features": int(feature_matrix.num_features),
        },
        "observed": observed,
    }

    if null_repeats <= 0:
        return result

    rng = np.random.default_rng(seed)

    # Label permutation null (for non-graph features)
    label_runs = []
    for repeat in range(null_repeats):
        order = rng.permutation(len(y))
        y_perm = [y[int(i)] for i in order]
        label_runs.append(_matrix_metrics(X, y_perm, seed=seed + 101 + repeat))
    result["null_controls"] = {"label_permutation": _aggregate_null(observed, label_runs)}

    return result


def subsample_dataset(dataset, n: int) -> Any:
    """Deterministically subsample a PatchEffectDataset to exactly n examples.

    Takes the first n examples per corruption slice, returns a new PatchEffectDataset.
    """
    from pig.patching import PatchEffectDataset

    # Count examples per corruption, take first n per slice
    counts: dict[str, int] = {}
    result = []
    for t in dataset:
        key = str(t.prompt_pair.slice_label)
        if key not in counts:
            counts[key] = 0
        if counts[key] < n:
            result.append(t)
            counts[key] += 1
    return PatchEffectDataset(tensors=result)


# ---- Ablation helpers ----

def _run_k_ablation(
    n: int,
    seed: int,
    k_grid: list[int],
    model,
    prompt_pairs: list,
    node_types: list[str],
    bootstrap_graphs_per_slice: int,
    sample_fraction: float,
    min_examples: int,
) -> dict[str, dict]:
    """Run k ablation: different k values for the same patch effects."""
    results = {}
    for k in k_grid:
        builder = create_graph_builder("correlation_topk", k=k, enforce_direction=True)
        # Recompute patch effects with this k (different k = different graphs, same patch effects)
        dataset = compute_patch_effects(
            model, prompt_pairs[:n], cache_dir=None, show_progress=False, node_types=node_types
        )
        bootstrap_graphs = build_bootstrap_slice_graphs(
            dataset, builder,
            graphs_per_slice=bootstrap_graphs_per_slice,
            sample_fraction=sample_fraction,
            min_examples=min_examples,
            seed=seed,
        )
        graphs_with_labels = bootstrap_graphs

        for feat_name, feat_builder in [
            ("fixed_layout_signed", compute_fixed_layout_features_from_list),
            ("wl_bootstrap", lambda g: compute_wl_features_from_list(g, depth=3)),
        ]:
            bl = _evaluate_baseline(
                graphs_with_labels,
                feature_name=f"{feat_name}_k{k}",
                feature_builder=feat_builder,
                seed=seed,
                null_repeats=10,
            )
            key = f"{feat_name}_k{k}"
            obs = bl.get("observed", {})
            results[key] = {
                "k": k,
                "acc_linear": obs.get("linear", {}).get("accuracy_mean"),
                "acc_rbf": obs.get("rbf", {}).get("accuracy_mean"),
                "feature_shape": bl.get("feature_shape"),
            }
    return results


def _run_wl_depth_ablation(
    n: int,
    seed: int,
    depth_grid: list[int],
    model,
    prompt_pairs: list,
    node_types: list[str],
    k: int = 5,
    bootstrap_graphs_per_slice: int = 32,
    sample_fraction: float = 0.75,
    min_examples: int = 3,
) -> dict[str, dict]:
    """Run WL depth ablation: same graphs, different WL depths."""
    results = {}
    builder = create_graph_builder("correlation_topk", k=k, enforce_direction=True)
    dataset = compute_patch_effects(
        model, prompt_pairs[:n], cache_dir=None, show_progress=False, node_types=node_types
    )
    bootstrap_graphs = build_bootstrap_slice_graphs(
        dataset, builder,
        graphs_per_slice=bootstrap_graphs_per_slice,
        sample_fraction=sample_fraction,
        min_examples=min_examples,
        seed=seed,
    )
    graphs_with_labels = bootstrap_graphs

    for depth in depth_grid:
        # Use a closure with default arg to avoid late binding
        feat_builder = lambda g, d=depth: compute_wl_features_from_list(g, depth=d)
        bl = _evaluate_baseline(
            graphs_with_labels,
            feature_name=f"wl_bootstrap_depth{depth}",
            feature_builder=feat_builder,
            seed=seed,
            null_repeats=10,
        )
        key = f"wl_bootstrap_depth{depth}"
        obs = bl.get("observed", {})
        results[key] = {
            "depth": depth,
            "acc_linear": obs.get("linear", {}).get("accuracy_mean"),
            "acc_rbf": obs.get("rbf", {}).get("accuracy_mean"),
            "feature_shape": bl.get("feature_shape"),
        }
    return results


def _run_direction_ablation(
    n: int,
    seed: int,
    model,
    prompt_pairs: list,
    node_types: list[str],
    k: int = 5,
    bootstrap_graphs_per_slice: int = 32,
    sample_fraction: float = 0.75,
    min_examples: int = 3,
) -> dict[str, dict]:
    """Run direction constraint ablation: enforce_direction vs not."""
    results = {}
    for enforce_dir in [True, False]:
        builder = create_graph_builder("correlation_topk", k=k, enforce_direction=enforce_dir)
        dataset = compute_patch_effects(
            model, prompt_pairs[:n], cache_dir=None, show_progress=False, node_types=node_types
        )
        bootstrap_graphs = build_bootstrap_slice_graphs(
            dataset, builder,
            graphs_per_slice=bootstrap_graphs_per_slice,
            sample_fraction=sample_fraction,
            min_examples=min_examples,
            seed=seed,
        )
        graphs_with_labels = bootstrap_graphs

        bl = _evaluate_baseline(
            graphs_with_labels,
            feature_name=f"fixed_layout_signed_dir{'enforced' if enforce_dir else 'free'}",
            feature_builder=compute_fixed_layout_features_from_list,
            seed=seed,
            null_repeats=10,
        )
        key = f"fixed_layout_signed_dir{'enforced' if enforce_dir else 'free'}"
        obs = bl.get("observed", {})
        results[key] = {
            "enforce_direction": enforce_dir,
            "acc_linear": obs.get("linear", {}).get("accuracy_mean"),
            "acc_rbf": obs.get("rbf", {}).get("accuracy_mean"),
            "feature_shape": bl.get("feature_shape"),
        }
    return results


# ---- Main experiment loop ----

def main() -> None:
    parser = argparse.ArgumentParser(description="Run NeurIPS revision experiments for PIG")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tasks", nargs="*", default=["ioi"],
                        help="Tasks to evaluate: ioi greater_than kv_retrieval")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated seeds")
    parser.add_argument("--n-grid", default=",".join(str(n) for n in DEFAULT_N_GRID),
                        help="Comma-separated n values for learning curve")
    parser.add_argument("--n-per-corruption", type=int, default=None,
                        help="Override n per corruption (uses n-grid max if not set)")
    parser.add_argument("--k", type=int, default=5, help="Top-k for graph builder")
    parser.add_argument("--node-types", default="res", help="Comma-separated node types")
    parser.add_argument("--null-repeats", type=int, default=10, help="Null control repeats")
    parser.add_argument("--bootstrap-graphs-per-slice", type=int, default=32)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--feature-sweep", nargs="*", default=None,
                        help="Features to evaluate")
    parser.add_argument("--hash-dim", type=int, default=1024)
    parser.add_argument("--coarse-buckets", type=int, default=105)
    parser.add_argument("--topk-k", type=int, default=10)
    parser.add_argument("--k-ablation-grid", default=None,
                        help="Comma-separated k values for k ablation")
    parser.add_argument("--wl-depth-grid", default=None,
                        help="Comma-separated WL depths for WL depth ablation")
    parser.add_argument("--direction-ablation", action="store_true",
                        help="Run direction constraint ablation")
    parser.add_argument("--dim-control", action="store_true",
                        help="Run dimensionality control (PCA + RandomProjection)")
    parser.add_argument("--patch-effect-raw", action="store_true",
                        help="Include patch_effect_raw in feature sweep")
    parser.add_argument("--kv-n-pairs", type=int, default=10,
                        help="Number of country-capital pairs for kv_retrieval")
    args = parser.parse_args()

    seeds = _parse_int_grid(args.seeds)
    n_grid = _parse_int_grid(args.n_grid)
    node_types = _parse_csv_values(args.node_types)
    max_n = max(n_grid)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs") / "neurips_revision" / f"run_{_utc_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = {
        "model_name": args.model_name,
        "device": args.device,
        "tasks": args.tasks,
        "seeds": seeds,
        "n_grid": n_grid,
        "k": args.k,
        "node_types": node_types,
        "null_repeats": args.null_repeats,
        "bootstrap_graphs_per_slice": args.bootstrap_graphs_per_slice,
        "feature_sweep": args.feature_sweep or ["default"],
        "hash_dim": args.hash_dim,
        "coarse_buckets": args.coarse_buckets,
        "topk_k": args.topk_k,
        "k_ablation_grid": args.k_ablation_grid,
        "wl_depth_grid": args.wl_depth_grid,
        "direction_ablation": args.direction_ablation,
        "dim_control": args.dim_control,
        "patch_effect_raw": args.patch_effect_raw,
        "kv_n_pairs": args.kv_n_pairs,
        "timestamp": _utc_stamp(),
        "commit": os.popen("cd /home/ruben/PIG && git rev-parse HEAD 2>/dev/null").read().strip(),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"=== NeurIPS Revision Study ===")
    print(f"Model: {args.model_name} | Seeds: {seeds} | n-grid: {n_grid}")
    print(f"Tasks: {args.tasks} | Output: {output_dir}")
    print()

    model = create_model(model_name=args.model_name, device=args.device)

    # Feature sweep mapping
    features_to_eval = set(args.feature_sweep) if args.feature_sweep else {"default"}
    has_default = "default" in features_to_eval
    has_dim_control = args.dim_control or "dim_control" in features_to_eval
    has_patch_raw = args.patch_effect_raw or "patch_effect_raw" in features_to_eval

    all_reports: list[dict] = []

    # Task config: each task defines how to generate prompts for binary classification
    TASK_CONFIG = {
        "ioi": {
            "generators": [
                ("ioi", {"corruption": "name_swap"}),
                ("ioi", {"corruption": "abba"}),
            ],
            "n_per_corruption": True,
        },
        "greater_than": {
            "generators": [("greater_than", {})],
            "n_per_corruption": False,
        },
        "kv_retrieval": {
            "generators": [("kv_retrieval", {"n_pairs": args.kv_n_pairs})],
            "n_per_corruption": False,
        },
    }

    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")

        config = TASK_CONFIG.get(task)
        if not config:
            print(f"Unknown task: {task}")
            continue

        generators = config["generators"]

        # Generate and validate prompts (once, with first seed)
        all_prompt_pairs = []
        for gen_name, gen_kwargs in generators:
            all_prompt_pairs.extend(_generate_task_pairs(gen_name, max_n, seeds[0], **gen_kwargs))
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2", cache_dir=None)
        failures = validate_y_star(tok, all_prompt_pairs)
        if failures:
            print(f"WARNING: {len(sum(failures.values(), []))} y_star are multi-token for task={task}")
        else:
            print(f"y_star validation passed for all {len(all_prompt_pairs)} pairs")

        for seed in seeds:
            print(f"\n  Seed {seed}...", end=" ", flush=True)

            # Generate prompts for this seed
            prompt_pairs = []
            for gen_name, gen_kwargs in generators:
                prompt_pairs.extend(_generate_task_pairs(gen_name, max_n, seed, **gen_kwargs))

            # Compute patch effects (always with max_n, used for all n values)
            cache_dir = None
            if args.cache_root:
                cache_dir = str(Path(args.cache_root) / f"{task}_seed{seed}_n{max_n}")

            print("computing patch effects...", end=" ", flush=True)
            dataset_full = compute_patch_effects(
                model, prompt_pairs, cache_dir=cache_dir, show_progress=True, node_types=node_types
            )
            print(f"({len(dataset_full)} tensors)", end=" ", flush=True)

            # Cache full dataset results
            full_results = {
                "task": task,
                "seed": seed,
                "n_max": max_n,
                "n_grid": n_grid,
                "k": args.k,
                "node_types": node_types,
                "num_tensors": len(dataset_full),
            }

            # ---- Subsample for each n in n_grid ----
            n_results: dict[int, dict] = {}

            for n in n_grid:
                dataset_n = subsample_dataset(dataset_full, n)

                # Build graphs
                builder = create_graph_builder("correlation_topk", k=args.k, enforce_direction=True)
                bootstrap_graphs = build_bootstrap_slice_graphs(
                    dataset_n, builder,
                    graphs_per_slice=args.bootstrap_graphs_per_slice,
                    sample_fraction=0.75,
                    min_examples=3,
                    seed=seed,
                )
                graphs_with_labels = bootstrap_graphs

                n_results[n] = {}

                # --- Default features ---
                if has_default:
                    for feat_name, feat_builder in [
                        ("fixed_layout_signed", compute_fixed_layout_features_from_list),
                        ("wl_bootstrap", lambda g: compute_wl_features_from_list(g, depth=3)),
                    ]:
                        bl = _evaluate_baseline(
                            graphs_with_labels,
                            feature_name=feat_name,
                            feature_builder=feat_builder,
                            seed=seed,
                            null_repeats=args.null_repeats,
                        )
                        n_results[n][feat_name] = bl

                    # Patch-effect features (raw tensors)
                    pe_bl = _evaluate_baseline_raw(
                        dataset_n,
                        feature_name="patch_effect_node_vector",
                        feature_builder=compute_patch_effect_features_from_list,
                        seed=seed,
                        null_repeats=args.null_repeats,
                    )
                    n_results[n]["patch_effect_node_vector"] = pe_bl

                    # Top-k node identity (raw tensors)
                    tk_bl = _evaluate_baseline_raw(
                        dataset_n,
                        feature_name=f"topk_node_k{args.topk_k}",
                        feature_builder=lambda t: compute_topk_node_features_from_list(t, k=args.topk_k),
                        seed=seed,
                        null_repeats=args.null_repeats,
                    )
                    n_results[n][f"topk_node_k{args.topk_k}"] = tk_bl

                    # Global histogram (bootstrap graphs)
                    hg_bl = _evaluate_baseline(
                        bootstrap_graphs,
                        feature_name="global_histogram",
                        feature_builder=compute_histogram_features_from_list,
                        seed=seed,
                        null_repeats=args.null_repeats,
                    )
                    n_results[n]["global_histogram"] = hg_bl

                    # Hashed fixed-layout
                    hash_bl = _evaluate_baseline(
                        graphs_with_labels,
                        feature_name=f"hashed_fixed_layout_d{args.hash_dim}",
                        feature_builder=lambda g: compute_hashed_fixed_layout_features(g, dim=args.hash_dim, seed=seed),
                        seed=seed,
                        null_repeats=args.null_repeats,
                    )
                    n_results[n][f"hashed_fixed_layout_d{args.hash_dim}"] = hash_bl

                    # Coarse count
                    coarse_bl = _evaluate_baseline(
                        graphs_with_labels,
                        feature_name=f"coarse_count_k{args.coarse_buckets}",
                        feature_builder=lambda g: compute_coarse_count_features(g, num_buckets=args.coarse_buckets),
                        seed=seed,
                        null_repeats=args.null_repeats,
                    )
                    n_results[n][f"coarse_count_k{args.coarse_buckets}"] = coarse_bl

                    # Label permutation null
                    lp_bl = _evaluate_baseline(
                        graphs_with_labels,
                        feature_name="label_permutation",
                        feature_builder=compute_label_permutation_features,
                        seed=seed,
                        null_repeats=args.null_repeats,
                    )
                    n_results[n]["label_permutation"] = lp_bl

                    # Random graph (matched degree)
                    rg_bl = _evaluate_baseline(
                        graphs_with_labels,
                        feature_name="random_graph_matched_degree",
                        feature_builder=lambda g: compute_random_graph_features(g, seed=seed),
                        seed=seed,
                        null_repeats=args.null_repeats,
                    )
                    n_results[n]["random_graph_matched_degree"] = rg_bl

                # --- Dim control (PCA + RandomProjection) ---
                if has_dim_control:
                    for dim_name, dim_target in [("pca96", 96), ("rp96", 96)]:
                        try:
                            if dim_name == "pca96":
                                from pig.graph_features import compute_pca_projected_features
                                feat_fn = lambda g, td=dim_target, s=seed: compute_pca_projected_features(g, target_dim=td, seed=s)
                            else:
                                from pig.graph_features import compute_random_projection_features
                                feat_fn = lambda g, td=dim_target, s=seed: compute_random_projection_features(g, target_dim=td, seed=s)
                            dim_bl = _evaluate_baseline(
                                graphs_with_labels,
                                feature_name=f"fixed_layout_signed_{dim_name}",
                                feature_builder=feat_fn,
                                seed=seed,
                                null_repeats=args.null_repeats,
                            )
                            n_results[n][f"fixed_layout_signed_{dim_name}"] = dim_bl
                        except Exception as e:
                            print(f"\n  PCA/ RP for n={n} FAILED: {type(e).__name__}: {e}", flush=True)
                            import traceback; traceback.print_exc()

                # --- Patch-effect raw (standalone) ---
                if has_patch_raw:
                    pe_raw_bl = _evaluate_baseline_raw(
                        dataset_n,
                        feature_name="patch_effect_raw",
                        feature_builder=compute_patch_effect_features_from_list,
                        seed=seed,
                        null_repeats=args.null_repeats,
                    )
                    n_results[n]["patch_effect_raw"] = pe_raw_bl

                print(f"n={n} done", end=" ", flush=True)

            # ---- Hyperparameter ablations (only at n=100) ----
            ablation_results: dict[str, dict] = {}

            if 100 in n_results:
                dataset_100 = subsample_dataset(dataset_full, 100)

                if args.k_ablation_grid:
                    k_grid = _parse_int_grid(args.k_ablation_grid)
                    print(f"\n  Running k ablation (k={k_grid})...", flush=True)
                    ablation_results["k_ablation"] = _run_k_ablation(
                        100, seed, k_grid, model, prompt_pairs, node_types,
                        args.bootstrap_graphs_per_slice, 0.75, 3,
                    )

                if args.wl_depth_grid:
                    depth_grid = _parse_int_grid(args.wl_depth_grid)
                    print(f"  Running WL depth ablation (depth={depth_grid})...", flush=True)
                    ablation_results["wl_depth_ablation"] = _run_wl_depth_ablation(
                        100, seed, depth_grid, model, prompt_pairs, node_types,
                        args.k, args.bootstrap_graphs_per_slice, 0.75, 3,
                    )

                if args.direction_ablation:
                    print("  Running direction ablation...", flush=True)
                    ablation_results["direction_ablation"] = _run_direction_ablation(
                        100, seed, model, prompt_pairs, node_types,
                        args.k, args.bootstrap_graphs_per_slice, 0.75, 3,
                    )

            # ---- Assemble report ----
            report = {
                "task": task,
                "seed": seed,
                "n_grid_results": n_results,
                "ablation_results": ablation_results,
            }

            # Save per-task per-seed JSON
            task_safe = task.replace(" ", "_")
            report_path = runs_dir / f"{task_safe}_seed{seed}.json"
            report_path.write_text(json.dumps(report, indent=2))

            # Print summary for this seed
            for n in n_grid:
                nr = n_results.get(n, {})
                lin_acc = nr.get("fixed_layout_signed", {}).get("observed", {}).get("linear", {}).get("accuracy_mean")
                wl_acc = nr.get("wl_bootstrap", {}).get("observed", {}).get("linear", {}).get("accuracy_mean")
                print(f"\n    n={n}: fl_lin={lin_acc:.4f} wl_lin={wl_acc:.4f}", flush=True)

            all_reports.append(report)

    # Write aggregated results
    aggregated = {
        "runs": all_reports,
        "config": config,
    }
    (output_dir / "results.json").write_text(json.dumps(aggregated, indent=2))

    print(f"\n{'='*60}")
    print(f"Done. Wrote {len(all_reports)} run reports to {output_dir}")
    print(f"Aggregated results: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
