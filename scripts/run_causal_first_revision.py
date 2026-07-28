#!/usr/bin/env python3
"""Causal-first revision runner for the paper v2 blockers.

Stages:
- di-candidate-validation: CI/PC/null edge groups with real causal eval.
- multislice-s10: S>=10 graph classification smoke/final benchmark.
- raw-vs-graph-budget: equal-budget raw and graph representation comparison.
- kernel-geometry: slice-centroid kernel diagnostics, gated on S>=10.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pig.causal import (  # noqa: E402
    CausalEvalConfig,
    evaluate_causal_edges,
    propose_causal_candidate_groups,
    split_discovery_evaluation_tensors,
)
from pig.graph import create_graph_builder  # noqa: E402
from pig.model import create_model  # noqa: E402
from pig.patching import PatchEffectDataset, PatchEffectTensor, compute_patch_effects  # noqa: E402
from pig.prompts import (  # noqa: E402
    ABBACorruption,
    IOIGenerator,
    PromptPair,
    SecondSubjectSwapCorruption,
    SliceLabel,
    create_induction_dataset,
)

from scripts.run_paper_decision_study import (  # noqa: E402
    MatrixBundle,
    _build_bootstrap_graphs,
    _fit_eval_svm,
    _fixed_layout_matrix,
    _graphlet_matrix,
    _hashed_fixed_layout_matrix,
    _null_test_accuracy,
    _patch_effect_matrix,
    _spectral_matrix,
    _split_dataset_by_slice,
    _surface_cue_matrix,
    _wl_matrix,
)


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def relabel_pair(pair: PromptPair, *, task: str, corruption: str) -> PromptPair:
    return replace(pair, slice_label=SliceLabel(task=task, corruption=corruption))


def make_s10_suite(n: int, seed: int, *, num_ioi_templates: int = 4) -> list[PromptPair]:
    """Build the default S=10 suite with balanced slice counts."""
    if num_ioi_templates < 4:
        raise ValueError("num_ioi_templates must be at least 4 for s10_v1")

    pairs: list[PromptPair] = []
    templates = IOIGenerator.TEMPLATES[:num_ioi_templates]
    corruptions = [
        ("abba", ABBACorruption()),
        ("second_subject_swap", SecondSubjectSwapCorruption()),
    ]
    for template_idx, template in enumerate(templates):
        for corr_idx, (corr_name, corruption) in enumerate(corruptions):
            generator = IOIGenerator(
                corruption=corruption,
                seed=seed + template_idx * 1_000 + corr_idx * 10_000,
                templates=[template],
            )
            task = f"ioi_t{template_idx}"
            pairs.extend(
                relabel_pair(pair, task=task, corruption=corr_name)
                for pair in generator.generate_batch(n)
            )

    pairs.extend(
        relabel_pair(pair, task="induction", corruption="token_swap")
        for pair in create_induction_dataset(
            n,
            corruption="token_swap",
            seed=seed + 90_000,
        )
    )
    pairs.extend(
        relabel_pair(pair, task="induction_late", corruption="token_swap")
        for pair in create_induction_dataset(
            n,
            corruption="token_swap_late",
            seed=seed + 100_000,
        )
    )
    return pairs


def suite_counts(pairs: Sequence[PromptPair]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        key = str(pair.slice_label)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def validate_s10_suite(pairs: Sequence[PromptPair], *, expected_per_slice: int) -> dict:
    counts = suite_counts(pairs)
    if len(counts) < 10:
        raise ValueError(f"s10_v1 requires at least 10 slices, found {len(counts)}")
    bad = {label: count for label, count in counts.items() if count != expected_per_slice}
    if bad:
        raise ValueError(f"s10_v1 is not balanced: {bad}")
    return {"num_slices": len(counts), "counts": counts}


def tail_align_tensor(tensor: PatchEffectTensor, *, tail_tokens: int) -> PatchEffectTensor:
    if tensor.num_tokens < tail_tokens:
        raise ValueError(
            f"Tensor for {tensor.prompt_pair.slice_label} has {tensor.num_tokens} "
            f"tokens, needs at least tail_tokens={tail_tokens}"
        )
    token_labels = tensor.token_labels[-tail_tokens:] if tensor.token_labels else []
    return replace(
        tensor,
        effects=np.asarray(tensor.effects[:, -tail_tokens:, :], dtype=np.float32),
        token_labels=token_labels,
    )


def tail_align_dataset(dataset: PatchEffectDataset, *, tail_tokens: int) -> PatchEffectDataset:
    aligned = PatchEffectDataset()
    for tensor in dataset:
        aligned.add(tail_align_tensor(tensor, tail_tokens=tail_tokens))
    return aligned


def load_or_compute_dataset(
    *,
    model,
    prompt_pairs: Sequence[PromptPair],
    cache_dir: Path,
    node_types: Sequence[str],
    tail_tokens: int | None,
    show_progress: bool,
) -> PatchEffectDataset:
    dataset = compute_patch_effects(
        model,
        prompt_pairs,
        cache_dir=str(cache_dir),
        show_progress=show_progress,
        node_types=tuple(node_types),
    )
    if tail_tokens is not None:
        dataset = tail_align_dataset(dataset, tail_tokens=tail_tokens)
    return dataset


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def flatten_dataset(dataset: PatchEffectDataset) -> list[PatchEffectTensor]:
    return list(dataset)


def evaluate_graph_seed(
    *,
    dataset: PatchEffectDataset,
    seed: int,
    graph_builder: str,
    k: int,
    graphs_per_slice: int,
    bootstrap_sample_fraction: float,
    bootstrap_min_examples: int,
    train_fraction: float,
    wl_depth: int,
    hashed_dim: int,
    spectral_values: int,
    null_repeats: int,
) -> dict:
    train_dataset, test_dataset, split_meta = _split_dataset_by_slice(
        dataset,
        train_fraction=train_fraction,
        seed=seed,
    )
    max_tokens = min(tensor.num_tokens for tensor in dataset)
    builder = create_graph_builder(graph_builder, k=k, enforce_direction=True)
    train_graphs = _build_bootstrap_graphs(
        train_dataset,
        builder,
        max_tokens=max_tokens,
        graphs_per_slice=graphs_per_slice,
        sample_fraction=bootstrap_sample_fraction,
        min_examples=bootstrap_min_examples,
        seed=seed + 10_000,
    )
    test_graphs = _build_bootstrap_graphs(
        test_dataset,
        builder,
        max_tokens=max_tokens,
        graphs_per_slice=graphs_per_slice,
        sample_fraction=bootstrap_sample_fraction,
        min_examples=bootstrap_min_examples,
        seed=seed + 20_000,
    )

    train_tensors = flatten_dataset(train_dataset)
    test_tensors = flatten_dataset(test_dataset)
    surface_train = _surface_cue_matrix(train_tensors)
    surface_test = _surface_cue_matrix(test_tensors, feature_names=surface_train.feature_names)
    raw_train = _patch_effect_matrix(train_tensors)
    raw_test = _patch_effect_matrix(test_tensors, feature_names=raw_train.feature_names)
    wl_train = _wl_matrix(train_graphs, depth=wl_depth)
    wl_test = _wl_matrix(test_graphs, depth=wl_depth, vocabulary=wl_train.feature_names)
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
    hashed_train = _hashed_fixed_layout_matrix(train_graphs, dim=hashed_dim, mode="sign")
    hashed_test = _hashed_fixed_layout_matrix(test_graphs, dim=hashed_dim, mode="sign")
    spectral_train = _spectral_matrix(train_graphs, num_values=spectral_values)
    spectral_test = _spectral_matrix(test_graphs, num_values=spectral_values)
    graphlet_train = _graphlet_matrix(train_graphs)
    graphlet_test = _graphlet_matrix(test_graphs)

    edge_shuffle = _null_test_accuracy(
        fixed_weighted_train,
        test_graphs,
        mode="weighted",
        kernel="linear",
        control="edge_shuffle",
        repeats=null_repeats,
        seed=seed + 30_000,
    )
    weight_shuffle = _null_test_accuracy(
        fixed_weighted_train,
        test_graphs,
        mode="weighted",
        kernel="linear",
        control="weight_shuffle",
        repeats=null_repeats,
        seed=seed + 40_000,
    )

    return {
        "seed": seed,
        "num_slices": len(dataset.get_slices()),
        "chance": 1.0 / len(dataset.get_slices()),
        "split": split_meta,
        "num_train_graphs": len(train_graphs),
        "num_test_graphs": len(test_graphs),
        "surface_cue_linear": _fit_eval_svm(surface_train, surface_test, kernel="linear", seed=seed)["accuracy"],
        "patch_effect_linear": _fit_eval_svm(raw_train, raw_test, kernel="linear", seed=seed)["accuracy"],
        "wl_linear": _fit_eval_svm(wl_train, wl_test, kernel="linear", seed=seed)["accuracy"],
        "fixed_sign_linear": _fit_eval_svm(fixed_sign_train, fixed_sign_test, kernel="linear", seed=seed)["accuracy"],
        "fixed_weighted_linear": _fit_eval_svm(fixed_weighted_train, fixed_weighted_test, kernel="linear", seed=seed)["accuracy"],
        "hashed_sign_linear": _fit_eval_svm(hashed_train, hashed_test, kernel="linear", seed=seed)["accuracy"],
        "spectral_linear": _fit_eval_svm(spectral_train, spectral_test, kernel="linear", seed=seed)["accuracy"],
        "graphlet_linear": _fit_eval_svm(graphlet_train, graphlet_test, kernel="linear", seed=seed)["accuracy"],
        "edge_shuffle_linear": edge_shuffle["accuracy_mean"],
        "weight_shuffle_linear": weight_shuffle["accuracy_mean"],
    }


def summarize_stage_rows(rows: Sequence[dict], metric_names: Sequence[str]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows]
        mean, std = mean_std(values)
        summary[metric] = {"mean": mean, "std": std}
    return summary


def run_di_candidate_validation(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / "di_candidate_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(args.model_name, device=args.device)
    prompt_pairs = []
    for corruption, offset in (("abba", 0), ("second_subject_swap", 10_000)):
        generator = IOIGenerator(
            corruption=ABBACorruption() if corruption == "abba" else SecondSubjectSwapCorruption(),
            seed=args.seed + offset,
        )
        prompt_pairs.extend(generator.generate_batch(args.n))

    dataset = load_or_compute_dataset(
        model=model,
        prompt_pairs=prompt_pairs,
        cache_dir=Path(args.cache_root) / "di_candidate_validation" / args.model_name,
        node_types=parse_csv_values(args.node_types),
        tail_tokens=args.tail_tokens,
        show_progress=args.show_progress,
    )
    tensors = flatten_dataset(dataset)
    discovery, evaluation, split_stats = split_discovery_evaluation_tensors(
        tensors,
        max_total_examples=min(args.di_examples, len(tensors)),
        seed=args.seed,
        discovery_fraction=0.5,
        return_stats=True,
    )
    groups = propose_causal_candidate_groups(
        discovery,
        num_edges=args.num_edges,
        node_types=tuple(parse_csv_values(args.node_types)),
        enforce_direction=True,
        eps=args.eps,
        seed=args.seed,
    )
    config = CausalEvalConfig(
        eps=args.eps,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
    )

    summary_rows = []
    report = {"split": split_stats, "groups": {}, "config": vars(args)}
    for group_name, candidates in groups.items():
        if not candidates:
            continue
        result = evaluate_causal_edges(
            model=model,
            prompt_pairs=[tensor.prompt_pair for tensor in evaluation],
            candidates=candidates,
            config=config,
            show_progress=args.show_progress,
            progress_desc=f"di-{group_name}",
        )
        result.run_metadata.update(
            {
                "candidate_group": group_name,
                "discovery_count": len(discovery),
                "evaluation_count": len(evaluation),
            }
        )
        result.save_json(output_dir / f"{group_name}.json")
        result.save_npz(output_dir / f"{group_name}.npz")

        i_values = [edge["level_a"]["I"]["mean"] for edge in result.edges]
        m_values = [edge["level_b"]["M"]["mean"] for edge in result.edges]
        necessity_values = [edge["level_c"]["necessity"]["mean"] for edge in result.edges]
        i_mean, i_std = mean_std(i_values)
        m_mean, m_std = mean_std(m_values)
        necessity_mean, necessity_std = mean_std(necessity_values)
        significant_i = sum(
            bool(edge["level_a"]["I"].get("significant_fdr_bh"))
            for edge in result.edges
        )
        significant_necessity = sum(
            bool(edge["level_c"]["necessity"].get("significant_fdr_bh"))
            for edge in result.edges
        )
        row = {
            "group": group_name,
            "num_edges": len(result.edges),
            "I_mean": i_mean,
            "I_std": i_std,
            "M_mean": m_mean,
            "M_std": m_std,
            "necessity_mean": necessity_mean,
            "necessity_std": necessity_std,
            "I_significant_fdr": significant_i,
            "necessity_significant_fdr": significant_necessity,
        }
        summary_rows.append(row)
        report["groups"][group_name] = row
        print(f"di row={row}", flush=True)

    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_multislice_s10(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / "multislice_s10"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(args.model_name, device=args.device)
    metric_names = [
        "surface_cue_linear",
        "patch_effect_linear",
        "wl_linear",
        "fixed_sign_linear",
        "fixed_weighted_linear",
        "hashed_sign_linear",
        "spectral_linear",
        "graphlet_linear",
        "edge_shuffle_linear",
        "weight_shuffle_linear",
    ]
    staged_report = {}

    for n in parse_csv_ints(args.n_grid):
        rows = []
        stage_dir = output_dir / f"n{n}_g{args.graphs_per_slice}"
        for seed in parse_csv_ints(args.seeds):
            print(f"s10 n={n} seed={seed}", flush=True)
            pairs = make_s10_suite(n, seed)
            suite_meta = validate_s10_suite(pairs, expected_per_slice=n)
            dataset = load_or_compute_dataset(
                model=model,
                prompt_pairs=pairs,
                cache_dir=Path(args.cache_root) / "multislice_s10" / args.model_name / f"n{n}" / f"seed{seed}",
                node_types=parse_csv_values(args.node_types),
                tail_tokens=args.tail_tokens,
                show_progress=args.show_progress,
            )
            row = evaluate_graph_seed(
                dataset=dataset,
                seed=seed,
                graph_builder=args.graph_builder,
                k=args.k,
                graphs_per_slice=args.graphs_per_slice,
                bootstrap_sample_fraction=args.bootstrap_sample_fraction,
                bootstrap_min_examples=args.bootstrap_min_examples,
                train_fraction=args.train_fraction,
                wl_depth=args.wl_depth,
                hashed_dim=args.hashed_dim,
                spectral_values=args.spectral_values,
                null_repeats=args.null_repeats,
            )
            flat_row = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            flat_row["n"] = n
            rows.append(flat_row)
            print(f"s10 row={flat_row}", flush=True)
            (stage_dir / f"seed{seed}_split.json").parent.mkdir(parents=True, exist_ok=True)
            (stage_dir / f"seed{seed}_split.json").write_text(
                json.dumps({"split": row["split"], "suite": suite_meta}, indent=2),
                encoding="utf-8",
            )

        write_csv(stage_dir / "summary.csv", rows)
        staged_report[f"n{n}_g{args.graphs_per_slice}"] = {
            "summary": summarize_stage_rows(rows, metric_names),
            "rows": rows,
        }

    (output_dir / f"staged_report_g{args.graphs_per_slice}.json").write_text(
        json.dumps(staged_report, indent=2),
        encoding="utf-8",
    )


def select_top_abs_columns(train: MatrixBundle, test: MatrixBundle, *, budget: int) -> tuple[MatrixBundle, MatrixBundle]:
    scores = np.mean(np.abs(train.X), axis=0)
    order = np.argsort(-scores)[: min(budget, train.X.shape[1])]
    names = [train.feature_names[int(idx)] for idx in order]
    return (
        MatrixBundle(X=train.X[:, order], y=train.y, feature_names=names),
        MatrixBundle(X=test.X[:, order], y=test.y, feature_names=names),
    )


def pca_bundle(train: MatrixBundle, test: MatrixBundle, *, budget: int, seed: int) -> tuple[MatrixBundle, MatrixBundle]:
    dim = min(budget, train.X.shape[0] - 1, train.X.shape[1])
    if dim <= 0:
        dim = 1
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.X)
    test_scaled = scaler.transform(test.X)
    pca = PCA(n_components=dim, random_state=seed)
    train_x = pca.fit_transform(train_scaled)
    test_x = pca.transform(test_scaled)
    names = [f"pca_{idx}" for idx in range(dim)]
    return MatrixBundle(train_x, train.y, names), MatrixBundle(test_x, test.y, names)


def random_projection_bundle(
    train: MatrixBundle,
    test: MatrixBundle,
    *,
    budget: int,
    seed: int,
) -> tuple[MatrixBundle, MatrixBundle]:
    rng = np.random.default_rng(seed)
    dim = min(budget, train.X.shape[1])
    projection = rng.normal(0.0, 1.0 / math.sqrt(max(dim, 1)), size=(train.X.shape[1], dim))
    names = [f"rp_{idx}" for idx in range(dim)]
    return (
        MatrixBundle(train.X @ projection, train.y, names),
        MatrixBundle(test.X @ projection, test.y, names),
    )


def run_raw_vs_graph_budget(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / "raw_vs_graph_budget"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(args.model_name, device=args.device)
    rows = []
    for seed in parse_csv_ints(args.seeds):
        pairs = make_s10_suite(args.n, seed)
        validate_s10_suite(pairs, expected_per_slice=args.n)
        dataset = load_or_compute_dataset(
            model=model,
            prompt_pairs=pairs,
            cache_dir=Path(args.cache_root) / "multislice_s10" / args.model_name / f"n{args.n}" / f"seed{seed}",
            node_types=parse_csv_values(args.node_types),
            tail_tokens=args.tail_tokens,
            show_progress=args.show_progress,
        )
        train_dataset, test_dataset, _ = _split_dataset_by_slice(
            dataset,
            train_fraction=args.train_fraction,
            seed=seed,
        )
        max_tokens = min(tensor.num_tokens for tensor in dataset)
        builder = create_graph_builder(args.graph_builder, k=args.k, enforce_direction=True)
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
        raw_train = _patch_effect_matrix(flatten_dataset(train_dataset))
        raw_test = _patch_effect_matrix(flatten_dataset(test_dataset), feature_names=raw_train.feature_names)
        fixed_train = _fixed_layout_matrix(train_graphs, mode="sign_topology")
        fixed_test = _fixed_layout_matrix(test_graphs, feature_names=fixed_train.feature_names, mode="sign_topology")
        wl_train = _wl_matrix(train_graphs, depth=args.wl_depth)
        wl_test = _wl_matrix(test_graphs, depth=args.wl_depth, vocabulary=wl_train.feature_names)
        spectral_train = _spectral_matrix(train_graphs, num_values=args.spectral_values)
        spectral_test = _spectral_matrix(test_graphs, num_values=args.spectral_values)
        graphlet_train = _graphlet_matrix(train_graphs)
        graphlet_test = _graphlet_matrix(test_graphs)

        full_pairs = [
            ("raw_full", raw_train, raw_test, raw_train.X.shape[1]),
            ("graph_fixed_sign_full", fixed_train, fixed_test, fixed_train.X.shape[1]),
            ("graph_wl_full", wl_train, wl_test, wl_train.X.shape[1]),
            ("graph_spectral", spectral_train, spectral_test, spectral_train.X.shape[1]),
            ("graph_graphlet", graphlet_train, graphlet_test, graphlet_train.X.shape[1]),
        ]
        for representation, train, test, dim in full_pairs:
            rows.append(
                {
                    "seed": seed,
                    "budget": dim,
                    "family": representation,
                    "accuracy": _fit_eval_svm(train, test, kernel="linear", seed=seed)["accuracy"],
                    "features": dim,
                }
            )

        for budget in parse_csv_ints(args.budgets):
            raw_top_train, raw_top_test = select_top_abs_columns(raw_train, raw_test, budget=budget)
            raw_pca_train, raw_pca_test = pca_bundle(raw_train, raw_test, budget=budget, seed=seed)
            raw_rp_train, raw_rp_test = random_projection_bundle(raw_train, raw_test, budget=budget, seed=seed)
            hashed_train = _hashed_fixed_layout_matrix(train_graphs, dim=budget, mode="sign")
            hashed_test = _hashed_fixed_layout_matrix(test_graphs, dim=budget, mode="sign")
            for family, train, test in (
                ("raw_top_abs", raw_top_train, raw_top_test),
                ("raw_pca", raw_pca_train, raw_pca_test),
                ("raw_random_projection", raw_rp_train, raw_rp_test),
                ("graph_hashed_sign", hashed_train, hashed_test),
            ):
                rows.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "family": family,
                        "accuracy": _fit_eval_svm(train, test, kernel="linear", seed=seed)["accuracy"],
                        "features": train.X.shape[1],
                    }
                )
    write_csv(output_dir / "summary.csv", rows)


def effective_rank(eigenvalues: np.ndarray) -> float:
    positive = np.asarray(eigenvalues, dtype=np.float64)
    positive = positive[positive > 1e-12]
    if positive.size == 0:
        return 0.0
    probs = positive / positive.sum()
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(np.exp(entropy))


def label_alignment(kernel: np.ndarray, labels: Sequence[str]) -> float:
    labels = list(labels)
    same = []
    different = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                same.append(float(kernel[i, j]))
            else:
                different.append(float(kernel[i, j]))
    if not same or not different:
        return float("nan")
    return float(np.mean(same) - np.mean(different))


def nearest_neighbor_purity(kernel: np.ndarray, labels: Sequence[str]) -> float:
    labels = list(labels)
    if len(labels) < 2:
        return float("nan")
    correct = 0
    for i in range(len(labels)):
        row = np.array(kernel[i], copy=True)
        row[i] = -np.inf
        nn = int(np.argmax(row))
        correct += int(labels[nn] == labels[i])
    return float(correct / len(labels))


def kernel_pca_coords(kernel: np.ndarray, *, n_components: int = 2) -> np.ndarray:
    n = kernel.shape[0]
    if n == 0:
        return np.zeros((0, n_components), dtype=np.float64)
    one = np.ones((n, n), dtype=np.float64) / n
    centered = kernel - one @ kernel - kernel @ one + one @ kernel @ one
    eigvals, eigvecs = np.linalg.eigh(centered)
    order = np.argsort(eigvals)[::-1][:n_components]
    selected_vals = np.maximum(eigvals[order], 0.0)
    coords = eigvecs[:, order] * np.sqrt(selected_vals)[None, :]
    if coords.shape[1] < n_components:
        coords = np.pad(coords, ((0, 0), (0, n_components - coords.shape[1])))
    return coords


def kernel_geometry_from_centroids(
    centroid_matrix: np.ndarray,
    slice_labels: Sequence[SliceLabel],
    *,
    seed: int,
    null_repeats: int,
) -> dict:
    if len(slice_labels) < 10:
        raise ValueError("kernel geometry requires at least 10 slices")
    scaled = StandardScaler().fit_transform(centroid_matrix)
    distances = pairwise_distances(scaled)
    nonzero = distances[distances > 1e-12]
    gamma = 1.0 / (2.0 * float(np.median(nonzero) ** 2)) if nonzero.size else 1.0
    kernels = {
        "linear": linear_kernel(scaled),
        "rbf": rbf_kernel(scaled, gamma=gamma),
    }
    tasks = [label.task for label in slice_labels]
    corruptions = [label.corruption for label in slice_labels]
    rng = np.random.default_rng(seed)
    report = {"num_slices": len(slice_labels), "gamma": gamma, "kernels": {}}
    for name, kernel in kernels.items():
        eigvals = np.linalg.eigvalsh(kernel)
        task_alignment = label_alignment(kernel, tasks)
        corruption_alignment = label_alignment(kernel, corruptions)
        task_null = []
        corruption_null = []
        for _ in range(null_repeats):
            task_null.append(label_alignment(kernel, list(rng.permutation(tasks))))
            corruption_null.append(label_alignment(kernel, list(rng.permutation(corruptions))))
        coords = kernel_pca_coords(kernel, n_components=2)
        report["kernels"][name] = {
            "effective_rank": effective_rank(eigvals),
            "spectral_entropy_rank": effective_rank(eigvals),
            "task_alignment": task_alignment,
            "task_alignment_null_mean": float(np.nanmean(task_null)),
            "task_alignment_exceeds_null": bool(task_alignment > float(np.nanmean(task_null))),
            "corruption_alignment": corruption_alignment,
            "corruption_alignment_null_mean": float(np.nanmean(corruption_null)),
            "corruption_alignment_exceeds_null": bool(corruption_alignment > float(np.nanmean(corruption_null))),
            "task_nn_purity": nearest_neighbor_purity(kernel, tasks),
            "corruption_nn_purity": nearest_neighbor_purity(kernel, corruptions),
            "kernel_pca": [
                {
                    "slice": str(label),
                    "task": label.task,
                    "corruption": label.corruption,
                    "x": float(coords[idx, 0]),
                    "y": float(coords[idx, 1]),
                }
                for idx, label in enumerate(slice_labels)
            ],
        }
    return report


def run_kernel_geometry(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / "kernel_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(args.model_name, device=args.device)
    pairs = make_s10_suite(args.n, args.seed)
    validate_s10_suite(pairs, expected_per_slice=args.n)
    dataset = load_or_compute_dataset(
        model=model,
        prompt_pairs=pairs,
        cache_dir=Path(args.cache_root) / "multislice_s10" / args.model_name / f"n{args.n}" / f"seed{args.seed}",
        node_types=parse_csv_values(args.node_types),
        tail_tokens=args.tail_tokens,
        show_progress=args.show_progress,
    )
    if len(dataset.get_slices()) < 10:
        raise ValueError("kernel geometry requires at least 10 slices")
    builder = create_graph_builder(args.graph_builder, k=args.k, enforce_direction=True)
    graphs = _build_bootstrap_graphs(
        dataset,
        builder,
        max_tokens=min(tensor.num_tokens for tensor in dataset),
        graphs_per_slice=args.graphs_per_slice,
        sample_fraction=args.bootstrap_sample_fraction,
        min_examples=args.bootstrap_min_examples,
        seed=args.seed + 10_000,
    )
    labels = sorted(dataset.get_slices(), key=str)

    representations = {
        "wl": _wl_matrix(graphs, depth=args.wl_depth),
        "fixed_weighted": _fixed_layout_matrix(graphs, mode="weighted"),
        "hashed_sign": _hashed_fixed_layout_matrix(graphs, dim=args.hashed_dim, mode="sign"),
        "spectral": _spectral_matrix(graphs, num_values=args.spectral_values),
        "graphlet": _graphlet_matrix(graphs),
    }
    report = {
        "num_slices": len(labels),
        "representations": {},
    }
    for name, matrix in representations.items():
        centroids = []
        for label in labels:
            indices = [idx for idx, graph_label in enumerate(matrix.y) if graph_label == label]
            centroids.append(np.mean(matrix.X[indices], axis=0))
        report["representations"][name] = kernel_geometry_from_centroids(
            np.stack(centroids),
            labels,
            seed=args.seed,
            null_repeats=args.null_repeats,
        )
    (output_dir / "kernel_geometry.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Causal-first paper revision runner")
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "di-candidate-validation",
            "multislice-s10",
            "raw-vs-graph-budget",
            "kernel-geometry",
        ],
    )
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--n-grid", default="20,50,100")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-edges", type=int, default=50)
    parser.add_argument("--di-examples", type=int, default=100)
    parser.add_argument("--node-types", default="res")
    parser.add_argument("--tail-tokens", type=int, default=9)
    parser.add_argument("--graph-builder", default="correlation_topk")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--graphs-per-slice", type=int, default=8)
    parser.add_argument("--bootstrap-sample-fraction", type=float, default=0.75)
    parser.add_argument("--bootstrap-min-examples", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--wl-depth", type=int, default=3)
    parser.add_argument("--hashed-dim", type=int, default=1024)
    parser.add_argument("--spectral-values", type=int, default=16)
    parser.add_argument("--null-repeats", type=int, default=3)
    parser.add_argument("--budgets", default="16,32,64,128,256,512,1024")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--permutation-samples", type=int, default=200)
    parser.add_argument(
        "--cache-root",
        default=".cache/paper_decision/causal_first_revision",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_decision/causal_first_revision",
    )
    parser.add_argument("--show-progress", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.stage == "di-candidate-validation":
        run_di_candidate_validation(args)
    elif args.stage == "multislice-s10":
        run_multislice_s10(args)
    elif args.stage == "raw-vs-graph-budget":
        run_raw_vs_graph_budget(args)
    elif args.stage == "kernel-geometry":
        run_kernel_geometry(args)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
