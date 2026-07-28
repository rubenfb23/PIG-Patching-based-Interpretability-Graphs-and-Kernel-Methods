#!/usr/bin/env python3
"""Build exhaustive GPT-2 head graphs and compare DI, CI, and path patching."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import multiprocessing as mp
from pathlib import Path

from pig.direct_influence import (
    FullGraphConfig,
    PromptGraphScores,
    aggregate_full_graph_scores,
    enumerate_forward_edges,
    enumerate_head_nodes,
    evaluate_prompt_full_graph,
    save_full_graph_result,
)
from pig.model import create_model
from pig.prompts import (
    PromptPair,
    create_ioi_dataset,
    get_prompt_pair_distractor,
)


def _parse_indices(raw: str) -> list[int] | None:
    value = raw.strip().lower()
    if value in {"", "all", "*"}:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _prompt_suite(
    model,
    corruptions: list[str],
    *,
    examples_per_corruption: int,
    seed: int,
    require_clean_better: bool,
    selection_pool_multiplier: int,
) -> tuple[list[PromptPair], dict]:
    prompts: list[PromptPair] = []
    report: dict[str, dict] = {}
    for index, corruption in enumerate(corruptions):
        candidate_count = max(
            examples_per_corruption,
            examples_per_corruption * selection_pool_multiplier,
        )
        candidates = create_ioi_dataset(
            n_examples=candidate_count,
            corruption=corruption,
            seed=seed + index * 10_000,
        )
        selected: list[PromptPair] = []
        seen: set[tuple[str, str]] = set()
        rejected_clean_not_better = 0
        for prompt in candidates:
            prompt_key = (prompt.x_cln, prompt.x_crp)
            if prompt_key in seen:
                continue
            seen.add(prompt_key)
            if require_clean_better:
                distractor = get_prompt_pair_distractor(prompt)
                clean_score = float(
                    model.score(
                        prompt.x_cln,
                        prompt.y_star,
                        distractor_token=distractor,
                    )
                )
                base_score = float(
                    model.score(
                        prompt.x_crp,
                        prompt.y_star,
                        distractor_token=distractor,
                    )
                )
                if clean_score <= base_score:
                    rejected_clean_not_better += 1
                    continue
            selected.append(prompt)
            if len(selected) >= examples_per_corruption:
                break

        if len(selected) < examples_per_corruption:
            raise RuntimeError(
                f"Only {len(selected)} valid unique prompts found for "
                f"{corruption}; requested {examples_per_corruption}. Increase "
                "--selection-pool-multiplier or disable --require-clean-better."
            )
        prompts.extend(selected)
        report[corruption] = {
            "candidate_pool": candidate_count,
            "unique_candidates_checked": len(seen),
            "selected": len(selected),
            "rejected_clean_not_better": rejected_clean_not_better,
        }
    return prompts, report


def _default_output_dir(model_name: str, seed: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    return Path("outputs/full_direct_influence") / f"{safe_model}_{stamp}_seed{seed}"


def _evaluate_shard_group(
    model_name: str,
    device: str,
    tasks: list[tuple[int, PromptPair, Path]],
    nodes,
    edges,
    eps: float,
    show_progress: bool,
) -> None:
    """Evaluate a stable prompt subset on one device and persist every shard."""
    print(f"[worker {device}] loading {model_name} for {len(tasks)} prompt(s)")
    model = create_model(model_name=model_name, device=device)
    for prompt_index, prompt, shard_path in tasks:
        print(f"[worker {device}] prompt {prompt_index}: {prompt.slice_label}")
        shard = evaluate_prompt_full_graph(
            model,
            prompt,
            nodes,
            edges,
            eps=eps,
            show_progress=show_progress,
        )
        shard.save(shard_path)
        print(f"[worker {device}] saved {shard_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the full cross-layer attention-head graph with direct "
            "influence, causal intervention, and path patching."
        )
    )
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Comma-separated devices for prompt-level parallelism "
            "(e.g. cuda:0,cuda:1,cuda:2,cuda:3)"
        ),
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated layer indices, or 'all' (default)",
    )
    parser.add_argument(
        "--heads",
        default="all",
        help="Comma-separated head indices, or 'all' (default)",
    )
    parser.add_argument(
        "--corruptions",
        default="name_swap",
        help=(
            "Comma-separated IOI corruption slices. The confirmatory default "
            "uses name_swap because GPT-2 must pass clean_score > base_score."
        ),
    )
    parser.add_argument(
        "--examples-per-corruption",
        type=int,
        default=2,
        help="Prompt pairs per corruption slice",
    )
    parser.add_argument(
        "--require-clean-better",
        dest="require_clean_better",
        action="store_true",
        default=True,
        help="Require clean target margin > corrupted target margin (default)",
    )
    parser.add_argument(
        "--no-require-clean-better",
        dest="require_clean_better",
        action="store_false",
        help="Keep prompts even when the clean target margin is not better",
    )
    parser.add_argument(
        "--selection-pool-multiplier",
        type=int,
        default=50,
        help="Candidate pool size multiplier for clean-better selection",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--ci-alpha", type=float, default=0.05)
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.0,
        help="Active iff the lower bootstrap bound exceeds this value",
    )
    parser.add_argument(
        "--edge-budget",
        type=int,
        default=256,
        help="Top positive edges used for density-matched sensitivity",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite-shards",
        action="store_true",
        help="Recompute prompt shards even when compatible files exist",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-source progress bars",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.examples_per_corruption <= 0:
        raise ValueError("--examples-per-corruption must be positive")
    if args.selection_pool_multiplier <= 0:
        raise ValueError("--selection-pool-multiplier must be positive")

    corruptions = [
        value.strip() for value in args.corruptions.split(",") if value.strip()
    ]
    if not corruptions:
        raise ValueError("--corruptions must include at least one slice")

    output_dir = args.output_dir or _default_output_dir(args.model_name, args.seed)
    shards_dir = output_dir / "prompt_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    devices = (
        [value.strip() for value in args.devices.split(",") if value.strip()]
        if args.devices
        else []
    )
    setup_device = devices[0] if devices else args.device
    print(f"Loading {args.model_name} on {setup_device or 'auto'}")
    model = create_model(model_name=args.model_name, device=setup_device)
    nodes = enumerate_head_nodes(
        model.n_layers,
        model.n_heads,
        layers=_parse_indices(args.layers),
        heads=_parse_indices(args.heads),
    )
    edges = enumerate_forward_edges(nodes)
    if not edges:
        raise ValueError("The selection must span at least two distinct layers")
    prompts, selection_report = _prompt_suite(
        model,
        corruptions,
        examples_per_corruption=args.examples_per_corruption,
        seed=args.seed,
        require_clean_better=args.require_clean_better,
        selection_pool_multiplier=args.selection_pool_multiplier,
    )

    print(
        f"Full universe: {len(nodes)} heads, {len(edges)} directed edges, "
        f"{len(prompts)} prompts"
    )
    print(
        "Prompt selection: "
        f"{'clean_score > base_score' if args.require_clean_better else 'unfiltered'} "
        f"{selection_report}"
    )
    prompt_scores: list[PromptGraphScores] = []
    expected_edge_ids = [edge.edge_id() for edge in edges]
    pending: list[tuple[int, PromptPair, Path]] = []
    for prompt_index, prompt in enumerate(prompts):
        shard_path = shards_dir / f"prompt_{prompt_index:04d}.npz"
        shard = None
        if shard_path.exists() and not args.overwrite_shards:
            candidate = PromptGraphScores.load(shard_path)
            same_prompt = (
                candidate.metadata.get("clean_prompt") == prompt.x_cln
                and candidate.metadata.get("corrupted_prompt") == prompt.x_crp
            )
            if candidate.edge_ids == expected_edge_ids and same_prompt:
                shard = candidate
                print(f"[{prompt_index + 1}/{len(prompts)}] resumed {shard_path}")

        if shard is None:
            pending.append((prompt_index, prompt, shard_path))

    if pending and devices:
        del model
        assignments = [pending[index :: len(devices)] for index in range(len(devices))]
        processes = []
        context = mp.get_context("spawn")
        for device, tasks in zip(devices, assignments):
            if not tasks:
                continue
            process = context.Process(
                target=_evaluate_shard_group,
                args=(
                    args.model_name,
                    device,
                    tasks,
                    nodes,
                    edges,
                    args.eps,
                    not args.no_progress,
                ),
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Prompt worker {process.pid} failed with exit code "
                    f"{process.exitcode}"
                )
    elif pending:
        for prompt_index, prompt, shard_path in pending:
            print(
                f"[{prompt_index + 1}/{len(prompts)}] "
                f"{prompt.slice_label}: exhaustive evaluation"
            )
            shard = evaluate_prompt_full_graph(
                model,
                prompt,
                nodes,
                edges,
                eps=args.eps,
                show_progress=not args.no_progress,
            )
            shard.save(shard_path)
            print(f"  saved {shard_path}")

    for prompt_index, prompt in enumerate(prompts):
        shard_path = shards_dir / f"prompt_{prompt_index:04d}.npz"
        shard = PromptGraphScores.load(shard_path)
        if shard.edge_ids != expected_edge_ids:
            raise RuntimeError(f"Incompatible edge universe in {shard_path}")
        if (
            shard.metadata.get("clean_prompt") != prompt.x_cln
            or shard.metadata.get("corrupted_prompt") != prompt.x_crp
        ):
            raise RuntimeError(f"Incompatible prompt content in {shard_path}")
        prompt_scores.append(shard)

    config = FullGraphConfig(
        eps=args.eps,
        bootstrap_samples=args.bootstrap_samples,
        ci_alpha=args.ci_alpha,
        active_threshold=args.active_threshold,
        edge_budget=args.edge_budget,
        seed=args.seed,
    )
    result = aggregate_full_graph_scores(
        prompt_scores,
        nodes,
        edges,
        config,
    )
    result["metadata"]["prompt_selection"] = {
        "require_clean_better": args.require_clean_better,
        "report": selection_report,
    }
    paths = save_full_graph_result(result, output_dir)

    print("\nPrimary structural comparison")
    print("method\treference\tprecision\trecall\tf1\tjaccard\tkernel_alignment")
    for row in result["summary_table"]:
        print(
            f"{row['method']}\t{row['reference']}\t"
            f"{row['precision']:.4f}\t{row['recall']:.4f}\t"
            f"{row['f1']:.4f}\t{row['jaccard']:.4f}\t"
            f"{row['kernel_alignment']:.4f}"
        )
    print("\nArtifacts")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
