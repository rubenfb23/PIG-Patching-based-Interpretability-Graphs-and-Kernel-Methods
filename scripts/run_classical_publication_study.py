#!/usr/bin/env python3
"""Run a reproducible classical PIG study for publication-oriented sweeps.

This script focuses on the classical publication path:

1. Generate IOI prompt pairs for the requested corruptions.
2. Compute patch effects for a chosen model and node-type set.
3. Build canonical slice graphs plus auxiliary per-example graphs.
4. Evaluate classical WL + SVM baselines with CV-safe preprocessing.
5. Run disjoint causal discovery/evaluation on the same tensors.
6. Write per-run JSON plus an aggregate CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from pig.causal import (
    CausalEvalConfig,
    build_clean_better_filter_report,
    evaluate_causal_edges,
    propose_causal_edge_candidates,
    split_discovery_evaluation_tensors,
)
from pig.embeddings import compute_wl_features_from_list
from pig.graph import build_bootstrap_slice_graphs, create_graph_builder
from pig.graph_features import (
    apply_null_control_to_graphs,
    compute_fixed_layout_features_from_list,
)
from pig.kernels import ClassicalKernelClassifier
from pig.model import create_model
from pig.patching import compute_patch_effects
from pig.prompts import create_ioi_dataset


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
    require_clean_better: bool,
) -> str:
    node_key = "-".join(node_types)
    filter_key = "clean_better" if require_clean_better else "all_examples"
    return f"seed{seed}_k{k}_nodes-{node_key}_n{num_examples}_{filter_key}"


def _slice_graph_stats(graphs: dict) -> dict[str, dict]:
    stats = {}
    for slice_label, graph in graphs.items():
        stats[str(slice_label)] = {
            **graph.compute_statistics(),
            "graph_role": graph.metadata.get("graph_role"),
            "construction_mode": graph.metadata.get("construction_mode"),
            "num_examples": graph.metadata.get("num_examples"),
        }
    return stats


def _matrix_classical_metrics(X, y, seed: int) -> dict[str, dict]:
    metrics = {}
    for kernel in ("linear", "rbf"):
        classifier = ClassicalKernelClassifier(
            kernel=kernel,
            random_state=seed,
        )
        cv_results = classifier.cross_validate(X, y)
        metrics[kernel] = cv_results
    return metrics


def _permuted_labels(labels, rng) -> list:
    order = rng.permutation(len(labels))
    return [labels[int(index)] for index in order]


def _aggregate_null_metrics(observed: dict, null_runs: list[dict]) -> dict:
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
        null_mean = float(sum(values) / len(values))
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


def _evaluate_feature_family(
    graphs_with_labels,
    *,
    feature_builder,
    seed: int,
    null_repeats: int,
) -> dict:
    if not graphs_with_labels:
        return {"error": "No graphs available for feature family"}

    feature_matrix = feature_builder(graphs_with_labels)
    X = feature_matrix.to_matrix()
    y = feature_matrix.slice_labels
    observed = _matrix_classical_metrics(X, y, seed=seed)

    result = {
        "feature_shape": {
            "num_graphs": int(feature_matrix.num_graphs),
            "num_features": int(feature_matrix.num_features),
        },
        "observed": observed,
        "null_controls": {},
    }
    if null_repeats <= 0:
        return result

    rng = np.random.default_rng(seed)
    label_runs = []
    for repeat in range(null_repeats):
        label_runs.append(
            _matrix_classical_metrics(
                X,
                _permuted_labels(y, rng),
                seed=seed + 101 + repeat,
            )
        )
    result["null_controls"]["label_permutation"] = _aggregate_null_metrics(
        observed,
        label_runs,
    )

    for control in ("edge_shuffle", "weight_shuffle"):
        graph_runs = []
        for repeat in range(null_repeats):
            null_graphs = apply_null_control_to_graphs(
                graphs_with_labels,
                control,
                seed=seed + 1_000 + repeat,
            )
            null_features = feature_builder(null_graphs)
            graph_runs.append(
                _matrix_classical_metrics(
                    null_features.to_matrix(),
                    null_features.slice_labels,
                    seed=seed + 2_000 + repeat,
                )
            )
        result["null_controls"][control] = _aggregate_null_metrics(
            observed,
            graph_runs,
        )

    return result


def _wl_feature_builder(depth: int):
    def build(graphs_with_labels):
        return compute_wl_features_from_list(graphs_with_labels, depth=depth)

    return build


def _fixed_layout_feature_builder(graphs_with_labels):
    return compute_fixed_layout_features_from_list(graphs_with_labels)


def _nested_metric(data: dict, path: Sequence[str], default=0.0):
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _causal_metrics(
    *,
    model,
    dataset,
    node_types: Sequence[str],
    seed: int,
    total_examples: int,
    num_edges: int,
    bootstrap_samples: int,
    permutation_samples: int,
    eps: float,
    require_clean_better: bool,
) -> dict:
    filter_report = build_clean_better_filter_report(
        dataset,
        bootstrap_samples=bootstrap_samples,
        ci_alpha=0.05,
        seed=seed,
    )
    if require_clean_better:
        eligible = [
            tensor for tensor in dataset if tensor.clean_score > tensor.base_score
        ]
        filter_rule = "clean_score > base_score"
    else:
        eligible = list(dataset)
        filter_rule = "no clean>base filter"
    split = split_discovery_evaluation_tensors(
        eligible,
        max_total_examples=min(total_examples, len(eligible)),
        seed=seed,
        return_stats=True,
    )
    discovery_tensors, evaluation_tensors, split_stats = split
    if not discovery_tensors or not evaluation_tensors:
        return {
            "filter_rule": filter_rule,
            "eligible_tensors": len(eligible),
            "require_clean_better": bool(require_clean_better),
            "clean_better_filter_report": filter_report,
            "split": split_stats,
            "error": split_stats.get("error", "Unable to form disjoint subsets"),
        }

    candidates = propose_causal_edge_candidates(
        discovery_tensors,
        num_edges=num_edges,
        node_types=node_types,
        enforce_direction=True,
        eps=eps,
    )
    if not candidates:
        return {
            "filter_rule": filter_rule,
            "eligible_tensors": len(eligible),
            "require_clean_better": bool(require_clean_better),
            "clean_better_filter_report": filter_report,
            "split": split_stats,
            "error": "No causal candidates proposed",
        }

    config = CausalEvalConfig(
        seed=seed,
        eps=eps,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
    )
    result = evaluate_causal_edges(
        model,
        [tensor.prompt_pair for tensor in evaluation_tensors],
        candidates,
        config,
        show_progress=False,
    )
    multiple_testing = result.run_metadata["multiple_testing"]["metrics"]
    i_fdr_significant = int(multiple_testing["I"]["significant_fdr_bh"])
    num_evaluated = int(result.run_metadata["num_edges_evaluated"])
    survival_rate = float(i_fdr_significant / num_evaluated) if num_evaluated else 0.0
    return {
        "filter_rule": filter_rule,
        "eligible_tensors": len(eligible),
        "require_clean_better": bool(require_clean_better),
        "clean_better_filter_report": filter_report,
        "split": split_stats,
        "num_candidates": len(candidates),
        "num_edges_evaluated": num_evaluated,
        "i_significant_fdr_bh": i_fdr_significant,
        "i_significant_bonferroni": int(
            multiple_testing["I"]["significant_bonferroni"]
        ),
        "i_survival_rate_fdr_bh": survival_rate,
        "multiple_testing": multiple_testing,
    }


def _write_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "run_key",
        "seed",
        "k",
        "node_types",
        "num_examples_per_corruption",
        "causal_filter_mode",
        "linear_accuracy_mean",
        "linear_accuracy_std",
        "rbf_accuracy_mean",
        "rbf_accuracy_std",
        "bootstrap_wl_linear_accuracy_mean",
        "bootstrap_wl_rbf_accuracy_mean",
        "fixed_layout_linear_accuracy_mean",
        "fixed_layout_rbf_accuracy_mean",
        "wl_per_example_linear_label_null_delta",
        "wl_bootstrap_linear_label_null_delta",
        "fixed_layout_linear_label_null_delta",
        "wl_per_example_linear_edge_null_delta",
        "wl_bootstrap_linear_edge_null_delta",
        "fixed_layout_linear_edge_null_delta",
        "wl_per_example_linear_weight_null_delta",
        "wl_bootstrap_linear_weight_null_delta",
        "fixed_layout_linear_weight_null_delta",
        "causal_i_survival_rate_fdr_bh",
        "causal_i_significant_fdr_bh",
        "causal_num_edges_evaluated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_classical_publication_study",
        description=(
            "Run a reproducible sweep for the classical PIG publication path."
        ),
    )
    parser.add_argument("--model-name", default="toy_transformer")
    parser.add_argument("--device", default=None)
    parser.add_argument("--corruptions", default="name_swap,abba")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--k-grid", default="1,3,5,10")
    parser.add_argument("--num-examples-grid", default="12,20")
    parser.add_argument(
        "--node-types-grid",
        default="res;res,mlp,att",
        help="Semicolon-separated node-type groups. Example: 'res;res,mlp,att'",
    )
    parser.add_argument("--graph-builder", default="correlation_topk")
    parser.add_argument("--wl-depth", type=int, default=3)
    parser.add_argument("--bootstrap-graphs-per-slice", type=int, default=12)
    parser.add_argument("--bootstrap-sample-fraction", type=float, default=0.75)
    parser.add_argument("--bootstrap-min-examples", type=int, default=3)
    parser.add_argument(
        "--null-repeats",
        type=int,
        default=5,
        help="Number of label/topology/weight null-control repeats per baseline.",
    )
    parser.add_argument("--causal-num-edges", type=int, default=5)
    parser.add_argument(
        "--causal-total-examples",
        type=int,
        default=20,
        help="Total examples used for disjoint causal discovery/evaluation.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--permutation-samples", type=int, default=200)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--causal-require-clean-better",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether causal discovery/evaluation should keep only tensors with "
            "clean_score > base_score. Use --no-causal-require-clean-better "
            "for sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Optional root directory for patch-effect caches, grouped by run key.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for study outputs "
            "(default: outputs/classical_publication/<model>_<utc>)."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

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
        / "classical_publication"
        / f"{args.model_name.replace('/', '_')}_{_utc_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(
        model_name=args.model_name,
        device=args.device,
    )

    summary_rows: list[dict] = []
    run_reports: list[dict] = []

    for seed in seeds:
        for k in k_grid:
            for node_types in node_type_grid:
                for num_examples in num_examples_grid:
                    run_key = _run_key(
                        seed=seed,
                        k=k,
                        node_types=node_types,
                        num_examples=num_examples,
                        require_clean_better=args.causal_require_clean_better,
                    )
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
                        show_progress=False,
                        node_types=node_types,
                    )

                    builder = create_graph_builder(
                        args.graph_builder,
                        k=k,
                        enforce_direction=True,
                    )
                    slice_graphs = builder.build_all(dataset)
                    example_graphs = builder.build_per_example(dataset)
                    bootstrap_graphs = build_bootstrap_slice_graphs(
                        dataset,
                        builder,
                        graphs_per_slice=args.bootstrap_graphs_per_slice,
                        sample_fraction=args.bootstrap_sample_fraction,
                        min_examples=args.bootstrap_min_examples,
                        seed=seed,
                    )

                    wl_builder = _wl_feature_builder(args.wl_depth)
                    auxiliary_baselines = {
                        "wl_per_example": _evaluate_feature_family(
                            example_graphs,
                            feature_builder=wl_builder,
                            seed=seed,
                            null_repeats=args.null_repeats,
                        ),
                        "wl_bootstrap_slice": _evaluate_feature_family(
                            bootstrap_graphs,
                            feature_builder=wl_builder,
                            seed=seed,
                            null_repeats=args.null_repeats,
                        ),
                        "fixed_layout_bootstrap_slice": _evaluate_feature_family(
                            bootstrap_graphs,
                            feature_builder=_fixed_layout_feature_builder,
                            seed=seed,
                            null_repeats=args.null_repeats,
                        ),
                    }
                    classical = auxiliary_baselines["wl_per_example"]["observed"]
                    causal = _causal_metrics(
                        model=model,
                        dataset=dataset,
                        node_types=node_types,
                        seed=seed,
                        total_examples=args.causal_total_examples,
                        num_edges=args.causal_num_edges,
                        bootstrap_samples=args.bootstrap_samples,
                        permutation_samples=args.permutation_samples,
                        eps=args.eps,
                        require_clean_better=args.causal_require_clean_better,
                    )

                    report = {
                        "run_key": run_key,
                        "model_name": args.model_name,
                        "seed": seed,
                        "k": k,
                        "node_types": list(node_types),
                        "num_examples_per_corruption": num_examples,
                        "corruptions": corruptions,
                        "slice_graphs": _slice_graph_stats(slice_graphs),
                        "classical_baselines": classical,
                        "auxiliary_baselines": auxiliary_baselines,
                        "causal_eval": causal,
                    }
                    run_reports.append(report)

                    with (runs_dir / f"{run_key}.json").open(
                        "w", encoding="utf-8"
                    ) as handle:
                        json.dump(report, handle, indent=2, sort_keys=True)

                    summary_rows.append(
                        {
                            "run_key": run_key,
                            "seed": seed,
                            "k": k,
                            "node_types": ",".join(node_types),
                            "num_examples_per_corruption": num_examples,
                            "causal_filter_mode": (
                                "clean_better"
                                if args.causal_require_clean_better
                                else "all_examples"
                            ),
                            "linear_accuracy_mean": classical["linear"][
                                "accuracy_mean"
                            ],
                            "linear_accuracy_std": classical["linear"]["accuracy_std"],
                            "rbf_accuracy_mean": classical["rbf"]["accuracy_mean"],
                            "rbf_accuracy_std": classical["rbf"]["accuracy_std"],
                            "bootstrap_wl_linear_accuracy_mean": _nested_metric(
                                auxiliary_baselines,
                                [
                                    "wl_bootstrap_slice",
                                    "observed",
                                    "linear",
                                    "accuracy_mean",
                                ],
                            ),
                            "bootstrap_wl_rbf_accuracy_mean": _nested_metric(
                                auxiliary_baselines,
                                [
                                    "wl_bootstrap_slice",
                                    "observed",
                                    "rbf",
                                    "accuracy_mean",
                                ],
                            ),
                            "fixed_layout_linear_accuracy_mean": _nested_metric(
                                auxiliary_baselines,
                                [
                                    "fixed_layout_bootstrap_slice",
                                    "observed",
                                    "linear",
                                    "accuracy_mean",
                                ],
                            ),
                            "fixed_layout_rbf_accuracy_mean": _nested_metric(
                                auxiliary_baselines,
                                [
                                    "fixed_layout_bootstrap_slice",
                                    "observed",
                                    "rbf",
                                    "accuracy_mean",
                                ],
                            ),
                            "wl_per_example_linear_label_null_delta": _nested_metric(
                                auxiliary_baselines,
                                [
                                    "wl_per_example",
                                    "null_controls",
                                    "label_permutation",
                                    "linear",
                                    "observed_minus_null_mean",
                                ],
                            ),
                            "wl_bootstrap_linear_label_null_delta": _nested_metric(
                                auxiliary_baselines,
                                [
                                    "wl_bootstrap_slice",
                                    "null_controls",
                                    "label_permutation",
                                    "linear",
                                    "observed_minus_null_mean",
                                ],
                            ),
                            "fixed_layout_linear_label_null_delta": _nested_metric(
                                auxiliary_baselines,
                                [
                                    "fixed_layout_bootstrap_slice",
                                    "null_controls",
                                    "label_permutation",
                                    "linear",
                                    "observed_minus_null_mean",
                                ],
                            ),
                            "wl_per_example_linear_edge_null_delta": _nested_metric(
                                auxiliary_baselines,
                                [
                                    "wl_per_example",
                                    "null_controls",
                                    "edge_shuffle",
                                    "linear",
                                    "observed_minus_null_mean",
                                ],
                            ),
                            "wl_bootstrap_linear_edge_null_delta": _nested_metric(
                                auxiliary_baselines,
                                [
                                    "wl_bootstrap_slice",
                                    "null_controls",
                                    "edge_shuffle",
                                    "linear",
                                    "observed_minus_null_mean",
                                ],
                            ),
                            "fixed_layout_linear_edge_null_delta": _nested_metric(
                                auxiliary_baselines,
                                [
                                    "fixed_layout_bootstrap_slice",
                                    "null_controls",
                                    "edge_shuffle",
                                    "linear",
                                    "observed_minus_null_mean",
                                ],
                            ),
                            "wl_per_example_linear_weight_null_delta": _nested_metric(
                                auxiliary_baselines,
                                [
                                    "wl_per_example",
                                    "null_controls",
                                    "weight_shuffle",
                                    "linear",
                                    "observed_minus_null_mean",
                                ],
                            ),
                            "wl_bootstrap_linear_weight_null_delta": _nested_metric(
                                auxiliary_baselines,
                                [
                                    "wl_bootstrap_slice",
                                    "null_controls",
                                    "weight_shuffle",
                                    "linear",
                                    "observed_minus_null_mean",
                                ],
                            ),
                            "fixed_layout_linear_weight_null_delta": _nested_metric(
                                auxiliary_baselines,
                                [
                                    "fixed_layout_bootstrap_slice",
                                    "null_controls",
                                    "weight_shuffle",
                                    "linear",
                                    "observed_minus_null_mean",
                                ],
                            ),
                            "causal_i_survival_rate_fdr_bh": causal.get(
                                "i_survival_rate_fdr_bh", 0.0
                            ),
                            "causal_i_significant_fdr_bh": causal.get(
                                "i_significant_fdr_bh", 0
                            ),
                            "causal_num_edges_evaluated": causal.get(
                                "num_edges_evaluated", 0
                            ),
                        }
                    )

    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": run_reports}, handle, indent=2, sort_keys=True)
    _write_summary_csv(output_dir / "summary.csv", summary_rows)

    print(f"Wrote {len(run_reports)} run reports to {output_dir}")


if __name__ == "__main__":
    main()
