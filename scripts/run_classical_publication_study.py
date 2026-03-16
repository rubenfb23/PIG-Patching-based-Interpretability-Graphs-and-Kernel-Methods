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
from typing import Iterable, Sequence

from pig.causal import (
    CausalEvalConfig,
    build_clean_better_filter_report,
    evaluate_causal_edges,
    propose_causal_edge_candidates,
    split_discovery_evaluation_tensors,
)
from pig.embeddings import compute_wl_features_from_list
from pig.graph import create_graph_builder
from pig.kernels import train_classical_baseline
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


def _classical_metrics(feature_matrix, seed: int) -> dict[str, dict]:
    metrics = {}
    for kernel in ("linear", "rbf"):
        _, cv_results = train_classical_baseline(
            feature_matrix,
            kernel=kernel,
            random_state=seed,
        )
        metrics[kernel] = cv_results
    return metrics


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
        eligible = [tensor for tensor in dataset if tensor.clean_score > tensor.base_score]
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
        "i_significant_bonferroni": int(multiple_testing["I"]["significant_bonferroni"]),
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
                    feature_matrix = compute_wl_features_from_list(
                        example_graphs,
                        depth=args.wl_depth,
                    )

                    classical = _classical_metrics(feature_matrix, seed=seed)
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
                        "causal_eval": causal,
                    }
                    run_reports.append(report)

                    with (runs_dir / f"{run_key}.json").open("w", encoding="utf-8") as handle:
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
                            "linear_accuracy_mean": classical["linear"]["accuracy_mean"],
                            "linear_accuracy_std": classical["linear"]["accuracy_std"],
                            "rbf_accuracy_mean": classical["rbf"]["accuracy_mean"],
                            "rbf_accuracy_std": classical["rbf"]["accuracy_std"],
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
