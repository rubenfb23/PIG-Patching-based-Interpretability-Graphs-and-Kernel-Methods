#!/usr/bin/env python3
"""Run an auxiliary GNN representation study on bootstrap slice graphs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from pig.gnn import GNNTrainingConfig, cross_validate_gnn_baseline
from pig.graph import build_bootstrap_slice_graphs, create_graph_builder
from pig.graph_features import apply_null_control_to_graphs
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
) -> str:
    node_key = "-".join(node_types)
    return f"seed{seed}_k{k}_nodes-{node_key}_n{num_examples}_clean_better"


def _aggregate_null_metrics(observed: dict, null_runs: list[dict]) -> dict:
    values = [float(run["accuracy_mean"]) for run in null_runs]
    null_mean = float(np.mean(values)) if values else 0.0
    return {
        "accuracy_mean": null_mean,
        "accuracy_std": float(np.std(values)) if values else 0.0,
        "observed_minus_null_mean": float(observed["accuracy_mean"] - null_mean),
        "repeats": len(values),
        "accuracy_values": values,
    }


def _permuted_labels(graphs_with_labels, rng):
    labels = [label for _, label in graphs_with_labels]
    order = rng.permutation(len(labels))
    return [
        (graph, labels[int(order[idx])])
        for idx, (graph, _) in enumerate(graphs_with_labels)
    ]


def _evaluate_gnn_family(
    graphs_with_labels,
    *,
    config: GNNTrainingConfig,
    null_repeats: int,
) -> dict:
    observed = cross_validate_gnn_baseline(
        graphs_with_labels,
        config=config,
    ).to_dict()
    result = {
        "observed": observed,
        "null_controls": {},
    }
    if null_repeats <= 0:
        return result

    rng = np.random.default_rng(config.random_state)
    label_runs = []
    for repeat in range(null_repeats):
        null_graphs = _permuted_labels(graphs_with_labels, rng)
        repeat_config = GNNTrainingConfig(
            **{**config.__dict__, "random_state": config.random_state + 101 + repeat}
        )
        label_runs.append(
            cross_validate_gnn_baseline(
                null_graphs,
                config=repeat_config,
            ).to_dict()
        )
    result["null_controls"]["label_permutation"] = _aggregate_null_metrics(
        observed,
        label_runs,
    )

    for control in ("edge_shuffle", "weight_shuffle"):
        control_runs = []
        for repeat in range(null_repeats):
            null_graphs = apply_null_control_to_graphs(
                graphs_with_labels,
                control,
                seed=config.random_state + 1_000 + repeat,
            )
            repeat_config = GNNTrainingConfig(
                **{
                    **config.__dict__,
                    "random_state": config.random_state + 2_000 + repeat,
                }
            )
            control_runs.append(
                cross_validate_gnn_baseline(
                    null_graphs,
                    config=repeat_config,
                ).to_dict()
            )
        result["null_controls"][control] = _aggregate_null_metrics(
            observed,
            control_runs,
        )

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_gnn_representation_study",
        description="Run GNN baselines on bootstrap slice PIG graphs.",
    )
    parser.add_argument("--model-name", default="toy_transformer")
    parser.add_argument("--device", default=None)
    parser.add_argument("--gnn-device", default="cpu")
    parser.add_argument("--corruptions", default="name_swap,abba")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--k-grid", default="5")
    parser.add_argument("--num-examples-grid", default="20")
    parser.add_argument("--node-types-grid", default="res")
    parser.add_argument("--graph-builder", default="correlation_topk")
    parser.add_argument("--bootstrap-graphs-per-slice", type=int, default=12)
    parser.add_argument("--bootstrap-sample-fraction", type=float, default=0.75)
    parser.add_argument("--bootstrap-min-examples", type=int, default=3)
    parser.add_argument("--null-repeats", type=int, default=5)
    parser.add_argument("--gnn-hidden-dim", type=int, default=32)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--gnn-dropout", type=float, default=0.1)
    parser.add_argument("--gnn-lr", type=float, default=1e-2)
    parser.add_argument("--gnn-weight-decay", type=float, default=1e-3)
    parser.add_argument("--gnn-epochs", type=int, default=150)
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
        / "gnn_representation"
        / f"{args.model_name.replace('/', '_')}_{_utc_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(args.model_name, device=args.device)
    summary_rows: list[dict] = []

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
                    bootstrap_graphs = build_bootstrap_slice_graphs(
                        dataset,
                        builder,
                        graphs_per_slice=args.bootstrap_graphs_per_slice,
                        sample_fraction=args.bootstrap_sample_fraction,
                        min_examples=args.bootstrap_min_examples,
                        seed=seed,
                    )
                    config = GNNTrainingConfig(
                        hidden_dim=args.gnn_hidden_dim,
                        num_layers=args.gnn_layers,
                        dropout=args.gnn_dropout,
                        lr=args.gnn_lr,
                        weight_decay=args.gnn_weight_decay,
                        epochs=args.gnn_epochs,
                        random_state=seed,
                        device=args.gnn_device,
                    )
                    gnn = _evaluate_gnn_family(
                        bootstrap_graphs,
                        config=config,
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
                        "gnn_bootstrap_slice": gnn,
                    }
                    with (runs_dir / f"{run_key}.json").open(
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        json.dump(report, handle, indent=2, sort_keys=True)

                    observed = gnn["observed"]
                    nulls = gnn["null_controls"]
                    summary_rows.append(
                        {
                            "run_key": run_key,
                            "seed": seed,
                            "accuracy_mean": observed["accuracy_mean"],
                            "accuracy_std": observed["accuracy_std"],
                            "label_null_delta": nulls.get("label_permutation", {}).get(
                                "observed_minus_null_mean", ""
                            ),
                            "edge_null_delta": nulls.get("edge_shuffle", {}).get(
                                "observed_minus_null_mean", ""
                            ),
                            "weight_null_delta": nulls.get("weight_shuffle", {}).get(
                                "observed_minus_null_mean", ""
                            ),
                        }
                    )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "run_key",
            "seed",
            "accuracy_mean",
            "accuracy_std",
            "label_null_delta",
            "edge_null_delta",
            "weight_null_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(summary_rows)} GNN run reports to {output_dir}")


if __name__ == "__main__":
    main()
