#!/usr/bin/env python3
"""Run a staged multi-slice GPT-2 smoke study.

This is intentionally narrower than the full paper-decision runner: it tests
whether a small multi-slice suite has graph-classifiable signal before scaling
to larger N. It reuses the example-disjoint split and bootstrap graph protocol
from ``run_paper_decision_study.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_paper_decision_study import (
    _build_bootstrap_graphs,
    _fit_eval_svm,
    _fixed_layout_matrix,
    _null_test_accuracy,
    _patch_effect_matrix,
    _split_dataset_by_slice,
    _surface_cue_matrix,
    _wl_matrix,
)

from pig.graph import create_graph_builder
from pig.model import create_model
from pig.patching import compute_patch_effects
from pig.prompts import (
    PromptPair,
    SliceLabel,
    create_greater_than_dataset,
    create_induction_dataset,
    create_ioi_dataset,
)


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _relabeled(pair: PromptPair, *, task: str, corruption: str) -> PromptPair:
    return replace(pair, slice_label=SliceLabel(task=task, corruption=corruption))


def _make_suite(n: int, seed: int, *, suite: str) -> list[PromptPair]:
    """Create a labelled prompt suite.

    ``ioi_induction4`` is deliberately conservative: two task families, four
    slices, and relatively short prompts. More ambitious suites should be added
    only after this pilot has non-trivial signal.
    """
    if suite not in {"ioi_induction4", "task_diverse4"}:
        raise ValueError(f"Unknown suite={suite!r}")

    pairs: list[PromptPair] = []
    if suite == "ioi_induction4":
        pairs.extend(create_ioi_dataset(n, corruption="abba", seed=seed))
        pairs.extend(
            create_ioi_dataset(n, corruption="second_subject_swap", seed=seed + 10_000)
        )
        pairs.extend(create_induction_dataset(n, corruption="token_swap", seed=seed + 20_000))
        pairs.extend(
            create_induction_dataset(n, corruption="token_swap_late", seed=seed + 30_000)
        )
    else:  # task_diverse4: three distinct task families
        pairs.extend(create_ioi_dataset(n, corruption="abba", seed=seed))
        pairs.extend(create_greater_than_dataset(n, seed=seed + 10_000))
        pairs.extend(create_induction_dataset(n, corruption="token_swap", seed=seed + 20_000))
        pairs.extend(
            create_induction_dataset(n, corruption="token_swap_late", seed=seed + 30_000)
        )
    # Keep the generator labels for IOI/induction, but normalize the second
    # induction corruption name to make printed tables unambiguous.
    normalized = []
    for pair in pairs:
        if pair.slice_label.task == "induction_late":
            normalized.append(
                _relabeled(pair, task="induction_late", corruption="token_swap")
            )
        else:
            normalized.append(pair)
    return normalized


def _mean_std(rows: Sequence[dict], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def _evaluate_stage(args: argparse.Namespace, n: int) -> tuple[list[dict], dict]:
    model = create_model(args.model_name, device=args.device)
    rows: list[dict] = []
    reports: dict = {}

    for seed in _parse_ints(args.seeds):
        print(f"stage n={n} seed={seed}", flush=True)
        prompt_pairs = _make_suite(n, seed, suite=args.suite)
        cache_dir = Path(args.cache_root) / args.suite / f"n{n}" / f"seed{seed}"
        dataset = compute_patch_effects(
            model,
            prompt_pairs,
            cache_dir=str(cache_dir),
            show_progress=True,
            node_types=("res",),
        )
        train_dataset, test_dataset, split_meta = _split_dataset_by_slice(
            dataset,
            train_fraction=args.train_fraction,
            seed=seed,
        )
        max_tokens = min(tensor.num_tokens for tensor in dataset)
        builder = create_graph_builder(
            args.graph_builder,
            k=args.k,
            enforce_direction=True,
        )
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

        wl_train = _wl_matrix(train_graphs, depth=args.wl_depth)
        wl_test = _wl_matrix(
            test_graphs,
            depth=args.wl_depth,
            vocabulary=wl_train.feature_names,
        )
        fixed_sign_train = _fixed_layout_matrix(train_graphs, mode="sign_topology")
        fixed_sign_test = _fixed_layout_matrix(
            test_graphs,
            feature_names=fixed_sign_train.feature_names,
            mode="sign_topology",
        )
        fixed_weighted_train = _fixed_layout_matrix(train_graphs, mode="weighted")
        fixed_weighted_test = _fixed_layout_matrix(
            test_graphs,
            feature_names=fixed_weighted_train.feature_names,
            mode="weighted",
        )
        raw_train = _patch_effect_matrix(list(train_dataset))
        raw_test = _patch_effect_matrix(
            list(test_dataset),
            feature_names=raw_train.feature_names,
        )
        surface_train = _surface_cue_matrix(list(train_dataset))
        surface_test = _surface_cue_matrix(
            list(test_dataset),
            feature_names=surface_train.feature_names,
        )

        edge_shuffle = _null_test_accuracy(
            fixed_weighted_train,
            test_graphs,
            mode="weighted",
            kernel="linear",
            control="edge_shuffle",
            repeats=args.null_repeats,
            seed=seed + 30_000,
        )
        weight_shuffle = _null_test_accuracy(
            fixed_weighted_train,
            test_graphs,
            mode="weighted",
            kernel="linear",
            control="weight_shuffle",
            repeats=args.null_repeats,
            seed=seed + 40_000,
        )

        row = {
            "n": n,
            "seed": seed,
            "num_slices": len(dataset.get_slices()),
            "chance": 1.0 / len(dataset.get_slices()),
            "max_tokens": max_tokens,
            "wl_linear": _fit_eval_svm(
                wl_train,
                wl_test,
                kernel="linear",
                seed=seed,
            )["accuracy"],
            "fixed_sign_linear": _fit_eval_svm(
                fixed_sign_train,
                fixed_sign_test,
                kernel="linear",
                seed=seed,
            )["accuracy"],
            "fixed_weighted_linear": _fit_eval_svm(
                fixed_weighted_train,
                fixed_weighted_test,
                kernel="linear",
                seed=seed,
            )["accuracy"],
            "patch_effect_linear": _fit_eval_svm(
                raw_train,
                raw_test,
                kernel="linear",
                seed=seed,
            )["accuracy"],
            "surface_cue_linear": _fit_eval_svm(
                surface_train,
                surface_test,
                kernel="linear",
                seed=seed,
            )["accuracy"],
            "edge_shuffle_linear": edge_shuffle["accuracy_mean"],
            "weight_shuffle_linear": weight_shuffle["accuracy_mean"],
        }
        rows.append(row)
        reports[f"seed{seed}"] = {
            "row": row,
            "split": split_meta,
            "slices": sorted(str(label) for label in dataset.get_slices()),
        }
        print(f"row={row}", flush=True)

    summary = {}
    for key in (
        "surface_cue_linear",
        "patch_effect_linear",
        "wl_linear",
        "fixed_sign_linear",
        "fixed_weighted_linear",
        "edge_shuffle_linear",
        "weight_shuffle_linear",
    ):
        mean, std = _mean_std(rows, key)
        summary[key] = {"mean": mean, "std": std}
    reports["summary"] = summary
    return rows, reports


def _write_stage(output_dir: Path, n: int, rows: list[dict], report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = output_dir / f"n{n}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    with (stage_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (stage_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _stage_passed(rows: list[dict]) -> bool:
    wl_mean, _ = _mean_std(rows, "wl_linear")
    fixed_mean, _ = _mean_std(rows, "fixed_sign_linear")
    surface_mean, _ = _mean_std(rows, "surface_cue_linear")
    chance = float(rows[0]["chance"])
    graph_mean = max(wl_mean, fixed_mean)
    return graph_mean >= chance + 0.15 and graph_mean >= surface_mean - 0.05


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Staged multi-slice smoke study")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--suite", default="ioi_induction4")
    parser.add_argument("--n-grid", default="20,50,100")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--graph-builder", default="correlation_topk")
    parser.add_argument("--graphs-per-slice", type=int, default=8)
    parser.add_argument("--bootstrap-sample-fraction", type=float, default=0.75)
    parser.add_argument("--bootstrap-min-examples", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--wl-depth", type=int, default=3)
    parser.add_argument("--null-repeats", type=int, default=3)
    parser.add_argument(
        "--cache-root",
        default=".cache/paper_decision/multislice_smoke",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_decision/gpt2_multislice_smoke_ioi_induction4_20260505",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop staged execution if graph results fail the smoke criterion.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    all_stage_reports = {}
    for n in _parse_ints(args.n_grid):
        rows, report = _evaluate_stage(args, n)
        _write_stage(output_dir, n, rows, report)
        passed = _stage_passed(rows)
        all_stage_reports[f"n{n}"] = {"passed": passed, **report}
        print(f"stage n={n} passed={passed}", flush=True)
        if args.stop_on_fail and not passed:
            break
    (output_dir / "staged_report.json").write_text(
        json.dumps(all_stage_reports, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
