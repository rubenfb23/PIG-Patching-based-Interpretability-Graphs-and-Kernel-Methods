#!/usr/bin/env python3
"""Run all paper experiments in a single reproducible script.

Usage:
    python scripts/run_paper_experiments.py --n 100 --seeds 7,42,123 --output-dir outputs/paper_run_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
)
from pig.kernels import ClassicalKernelClassifier
from pig.model import create_model
from pig.patching import compute_patch_effects
from pig.prompts import create_ioi_dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_grid(raw: str) -> list[int]:
    return [int(part) for part in _parse_csv_values(raw)]


def _matrix_metrics(X, y, seed: int) -> dict:
    metrics = {}
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

    return result


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all paper experiments")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--corruptions", default="name_swap,abba")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n", type=int, default=100, help="Prompt pairs per corruption")
    parser.add_argument("--node-types", default="res", help="Comma-separated node types")
    parser.add_argument("--null-repeats", type=int, default=10, help="Null control repeats")
    parser.add_argument("--causal-num-edges", type=int, default=20, help="Number of causal edges to evaluate")
    parser.add_argument("--causal-total-examples", type=int, default=40, help="Examples for causal discovery/eval")
    parser.add_argument("--bootstrap-graphs-per-slice", type=int, default=32)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--feature-sweep", nargs="*", default=["fixed_layout", "wl_bootstrap", "patch_effect", "topk_node", "histogram", "hash", "coarse_count"],
                        help="Which features to evaluate: fixed_layout wl_bootstrap wl_per_example patch_effect topk_node histogram hash coarse_count label_permutation random_graph")
    parser.add_argument("--hash-dim", type=int, default=1024, help="Hashed feature dimension")
    parser.add_argument("--coarse-buckets", type=int, default=105, help="Coarse count bucket count")
    parser.add_argument("--topk-k", type=int, default=10, help="Top-k node features")
    args = parser.parse_args()

    seeds = _parse_int_grid(args.seeds)
    corruptions = _parse_csv_values(args.corruptions)
    node_types = _parse_csv_values(args.node_types)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs") / "paper_run" / f"v{_utc_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = {
        "model_name": args.model_name,
        "device": args.device,
        "corruptions": corruptions,
        "seeds": seeds,
        "k": args.k,
        "n": args.n,
        "node_types": node_types,
        "null_repeats": args.null_repeats,
        "causal_num_edges": args.causal_num_edges,
        "causal_total_examples": args.causal_total_examples,
        "bootstrap_graphs_per_slice": args.bootstrap_graphs_per_slice,
        "feature_sweep": args.feature_sweep,
        "hash_dim": args.hash_dim,
        "coarse_buckets": args.coarse_buckets,
        "timestamp": _utc_stamp(),
        "commit": os.popen("cd /home/ruben/PIG && git rev-parse HEAD 2>/dev/null").read().strip(),
    }
    with (output_dir / "config.json").write_text(json.dumps(config, indent=2)) as _:
        pass

    model = create_model(model_name=args.model_name, device=args.device)

    all_rows: list[dict] = []
    all_reports: list[dict] = []

    for seed in seeds:
        # Generate prompts
        prompt_pairs = []
        for corruption in corruptions:
            prompt_pairs.extend(
                create_ioi_dataset(n_examples=args.n, corruption=corruption, seed=seed)
            )

        # Compute patch effects
        cache_dir = None
        if args.cache_root:
            cache_dir = str(Path(args.cache_root) / f"seed{seed}_n{args.n}")

        dataset = compute_patch_effects(
            model, prompt_pairs, cache_dir=cache_dir, show_progress=True, node_types=node_types
        )

        # Build graphs
        builder = create_graph_builder("correlation_topk", k=args.k, enforce_direction=True)
        slice_graphs = builder.build_all(dataset)
        example_graphs = builder.build_per_example(dataset)
        bootstrap_graphs = build_bootstrap_slice_graphs(
            dataset, builder,
            graphs_per_slice=args.bootstrap_graphs_per_slice,
            sample_fraction=0.75,
            min_examples=3,
            seed=seed,
        )

        features_to_eval = set(args.feature_sweep)
        report = {
            "seed": seed,
            "n": args.n,
            "k": args.k,
            "model_name": args.model_name,
            "node_types": node_types,
            "num_graphs": {
                "slice": len(slice_graphs),
                "per_example": len(example_graphs),
                "bootstrap": len(bootstrap_graphs),
            },
        }

        # --- Fixed-layout (bootstrap slice graphs) ---
        if "fixed_layout" in features_to_eval:
            bl = _evaluate_baseline(
                bootstrap_graphs,
                feature_name="fixed_layout_signed",
                feature_builder=compute_fixed_layout_features_from_list,
                seed=seed,
                null_repeats=args.null_repeats,
            )
            report["fixed_layout"] = bl
            observed_lin = bl.get("observed", {}).get("linear", {})
            all_rows.append({
                "seed": seed, "n": args.n, "k": args.k, "model": args.model_name,
                "feature": "fixed_layout_signed", "acc_linear": observed_lin.get("accuracy_mean"),
                "acc_rbf": bl.get("observed", {}).get("rbf", {}).get("accuracy_mean"),
            })

        # --- WL subtree (bootstrap slice graphs) ---
        if "wl_bootstrap" in features_to_eval:
            wl_builder = lambda g: compute_wl_features_from_list(g, depth=3)
            bl = _evaluate_baseline(
                bootstrap_graphs,
                feature_name="wl_bootstrap",
                feature_builder=wl_builder,
                seed=seed,
                null_repeats=args.null_repeats,
            )
            report["wl_bootstrap"] = bl
            observed_lin = bl.get("observed", {}).get("linear", {})
            all_rows.append({
                "seed": seed, "n": args.n, "k": args.k, "model": args.model_name,
                "feature": "wl_bootstrap", "acc_linear": observed_lin.get("accuracy_mean"),
                "acc_rbf": bl.get("observed", {}).get("rbf", {}).get("accuracy_mean"),
            })

        # --- Patch-effect node vector (no graph) ---
        if "patch_effect" in features_to_eval:
            bl = _evaluate_baseline(
                dataset,
                feature_name="patch_effect",
                feature_builder=compute_patch_effect_features_from_list,
                seed=seed,
                null_repeats=args.null_repeats,
            )
            report["patch_effect"] = bl
            observed_lin = bl.get("observed", {}).get("linear", {})
            all_rows.append({
                "seed": seed, "n": args.n, "k": args.k, "model": args.model_name,
                "feature": "patch_effect_node_vector", "acc_linear": observed_lin.get("accuracy_mean"),
                "acc_rbf": bl.get("observed", {}).get("rbf", {}).get("accuracy_mean"),
            })

        # --- Top-k node identity features ---
        if "topk_node" in features_to_eval:
            bl = _evaluate_baseline(
                dataset,
                feature_name=f"topk_node_k{args.topk_k}",
                feature_builder=lambda t: compute_topk_node_features_from_list(t, k=args.topk_k),
                seed=seed,
                null_repeats=args.null_repeats,
            )
            report["topk_node"] = bl
            observed_lin = bl.get("observed", {}).get("linear", {})
            all_rows.append({
                "seed": seed, "n": args.n, "k": args.k, "model": args.model_name,
                "feature": f"topk_node_k{args.topk_k}", "acc_linear": observed_lin.get("accuracy_mean"),
                "acc_rbf": bl.get("observed", {}).get("rbf", {}).get("accuracy_mean"),
            })

        # --- Global histogram features ---
        if "histogram" in features_to_eval:
            bl = _evaluate_baseline(
                [(g, l) for l, g in slice_graphs.items()],
                feature_name="global_histogram",
                feature_builder=compute_histogram_features_from_list,
                seed=seed,
                null_repeats=args.null_repeats,
            )
            report["histogram"] = bl
            observed_lin = bl.get("observed", {}).get("linear", {})
            all_rows.append({
                "seed": seed, "n": args.n, "k": args.k, "model": args.model_name,
                "feature": "global_histogram", "acc_linear": observed_lin.get("accuracy_mean"),
                "acc_rbf": bl.get("observed", {}).get("rbf", {}).get("accuracy_mean"),
            })

        # --- Hashed fixed-layout ---
        if "hash" in features_to_eval:
            bl = _evaluate_baseline(
                bootstrap_graphs,
                feature_name=f"hashed_fixed_layout_d{args.hash_dim}",
                feature_builder=lambda g: compute_hashed_fixed_layout_features(g, dim=args.hash_dim, seed=seed),
                seed=seed,
                null_repeats=args.null_repeats,
            )
            report["hash"] = bl
            observed_lin = bl.get("observed", {}).get("linear", {})
            all_rows.append({
                "seed": seed, "n": args.n, "k": args.k, "model": args.model_name,
                "feature": f"hashed_fixed_layout_d{args.hash_dim}", "acc_linear": observed_lin.get("accuracy_mean"),
                "acc_rbf": bl.get("observed", {}).get("rbf", {}).get("accuracy_mean"),
            })

        # --- Coarse count ---
        if "coarse_count" in features_to_eval:
            bl = _evaluate_baseline(
                bootstrap_graphs,
                feature_name=f"coarse_count_k{args.coarse_buckets}",
                feature_builder=lambda g: compute_coarse_count_features(g, num_buckets=args.coarse_buckets),
                seed=seed,
                null_repeats=args.null_repeats,
            )
            report["coarse_count"] = bl
            observed_lin = bl.get("observed", {}).get("linear", {})
            all_rows.append({
                "seed": seed, "n": args.n, "k": args.k, "model": args.model_name,
                "feature": f"coarse_count_k{args.coarse_buckets}", "acc_linear": observed_lin.get("accuracy_mean"),
                "acc_rbf": bl.get("observed", {}).get("rbf", {}).get("accuracy_mean"),
            })

        # --- Label permutation null ---
        if "label_permutation" in features_to_eval:
            bl = _evaluate_baseline(
                bootstrap_graphs,
                feature_name="label_permutation",
                feature_builder=compute_label_permutation_features,
                seed=seed,
                null_repeats=args.null_repeats,
            )
            report["label_permutation"] = bl
            observed_lin = bl.get("observed", {}).get("linear", {})
            all_rows.append({
                "seed": seed, "n": args.n, "k": args.k, "model": args.model_name,
                "feature": "label_permutation", "acc_linear": observed_lin.get("accuracy_mean"),
                "acc_rbf": bl.get("observed", {}).get("rbf", {}).get("accuracy_mean"),
            })

        # --- Random graph (matched degree) ---
        if "random_graph" in features_to_eval:
            bl = _evaluate_baseline(
                bootstrap_graphs,
                feature_name="random_graph_matched_degree",
                feature_builder=lambda g: compute_random_graph_features(g, seed=seed),
                seed=seed,
                null_repeats=args.null_repeats,
            )
            report["random_graph"] = bl
            observed_lin = bl.get("observed", {}).get("linear", {})
            all_rows.append({
                "seed": seed, "n": args.n, "k": args.k, "model": args.model_name,
                "feature": "random_graph_matched_degree", "acc_linear": observed_lin.get("accuracy_mean"),
                "acc_rbf": bl.get("observed", {}).get("rbf", {}).get("accuracy_mean"),
            })

        # --- WL per-example ---
        if "wl_per_example" in features_to_eval:
            wl_builder = lambda g: compute_wl_features_from_list(g, depth=3)
            bl = _evaluate_baseline(
                example_graphs,
                feature_name="wl_per_example",
                feature_builder=wl_builder,
                seed=seed,
                null_repeats=args.null_repeats,
            )
            report["wl_per_example"] = bl
            observed_lin = bl.get("observed", {}).get("linear", {})
            all_rows.append({
                "seed": seed, "n": args.n, "k": args.k, "model": args.model_name,
                "feature": "wl_per_example", "acc_linear": observed_lin.get("accuracy_mean"),
                "acc_rbf": bl.get("observed", {}).get("rbf", {}).get("accuracy_mean"),
            })

        # --- Causal validation (top-ranked vs random vs low-rank) ---
        filter_report = build_clean_better_filter_report(dataset, seed=seed)
        eligible = [t for t in dataset if t.clean_score > t.base_score]
        split = split_discovery_evaluation_tensors(
            eligible, max_total_examples=args.causal_total_examples, seed=seed
        )
        discovery_tensors, evaluation_tensors = split
        if discovery_tensors and evaluation_tensors:
            eval_config = CausalEvalConfig(
                seed=seed, bootstrap_samples=200, permutation_samples=200
            )
            causal_results = {}

            for label, propose_fn in [
                ("proposed", propose_causal_edge_candidates),
                ("random", propose_random_edge_candidates),
                ("low_rank", propose_lowrank_edge_candidates),
            ]:
                candidates = propose_fn(
                    discovery_tensors, num_edges=args.causal_num_edges,
                    node_types=("att",), enforce_direction=True,
                    seed=seed if label == "proposed" else seed + 100,
                )
                if candidates:
                    result = evaluate_causal_edges(
                        model,
                        [t.prompt_pair for t in evaluation_tensors],
                        candidates,
                        eval_config,
                        show_progress=False,
                    )
                    causal_results[label] = {
                        "num_candidates": len(candidates),
                        "num_edges_evaluated": len(result.edges),
                        "num_prompt_pairs_used": result.run_metadata["num_prompt_pairs_used"],
                        "edge_summaries": [
                            {
                                "edge_id": e["edge_id"],
                                "I_mean": e["level_a"]["I"]["mean"],
                                "M_mean": e["level_b"]["M"]["mean"],
                                "classification": e["diagnostics"]["classification"],
                                "rank": e["candidate"]["rank"],
                            }
                            for e in result.edges
                        ],
                    }

            if causal_results:
                report["causal_eval"] = causal_results

        all_reports.append(report)
        with (runs_dir / f"seed{seed}_n{args.n}.json").write_text(
            json.dumps(report, indent=2)
        ) as _:
            pass

    # Write summary
    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    with (output_dir / "results.json").write_text(
        json.dumps({"runs": all_reports}, indent=2)
    ) as _:
        pass

    print(f"Done. Wrote {len(all_reports)} run reports to {output_dir}")


if __name__ == "__main__":
    main()
