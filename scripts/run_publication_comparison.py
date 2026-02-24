#!/usr/bin/env python3
"""Run publication-grade PIG pipelines for base and fine-tuned models."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import os
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _model_key(model_name: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name.strip())
    key = key.strip("._-")
    return key.lower() or "model"


def _parse_seed_csv(value: str) -> list[int]:
    seeds: list[int] = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        seeds.append(int(stripped))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _build_pipeline_command(
    *,
    uv_bin: str,
    model_name: str,
    graph_builder: str,
    pipeline_log_path: Path,
    causal_cache_dir: Path,
    causal_output_dir: Path,
    causal_seed: int,
    causal_num_examples: int,
    causal_num_edges: int,
    causal_node_types: str,
    causal_bootstrap_samples: int,
    causal_permutation_samples: int,
    causal_build_cache_num_examples: int,
    causal_build_cache_corruptions: str,
    causal_device: str | None,
    continue_on_error: bool,
    causal_overwrite: bool,
) -> list[str]:
    command = [
        uv_bin,
        "run",
        "pig",
        "pipeline",
        "--model-name",
        model_name,
        "--graph-builder",
        graph_builder,
        "--pipeline-log-path",
        str(pipeline_log_path),
        "--with-causal-eval",
        "--causal-cache-dir",
        str(causal_cache_dir),
        "--causal-build-cache",
        "--causal-build-cache-num-examples",
        str(causal_build_cache_num_examples),
        "--causal-build-cache-corruptions",
        causal_build_cache_corruptions,
        "--causal-build-cache-node-types",
        causal_node_types,
        "--causal-output-dir",
        str(causal_output_dir),
        "--causal-seed",
        str(causal_seed),
        "--causal-num-examples",
        str(causal_num_examples),
        "--causal-num-edges",
        str(causal_num_edges),
        "--causal-node-types",
        causal_node_types,
        "--causal-bootstrap-samples",
        str(causal_bootstrap_samples),
        "--causal-permutation-samples",
        str(causal_permutation_samples),
    ]
    if causal_device:
        command.extend(["--causal-device", causal_device])
    if continue_on_error:
        command.append("--continue-on-error")
    if causal_overwrite:
        command.append("--causal-overwrite")
    return command


_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _run_command(
    command: list[str], cwd: Path, dry_run: bool, env: dict | None = None
) -> int:
    _log(f"$ {shlex.join(command)}")
    if dry_run:
        return 0
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(command, cwd=str(cwd), check=False, env=merged_env)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run sequential publication-grade pipeline jobs for GPT-2 base and "
            "a fine-tuned checkpoint."
        )
    )
    parser.add_argument(
        "--base-model",
        default="gpt2",
        help="Base model name/path for the first run (default: gpt2)",
    )
    parser.add_argument(
        "--finetuned-model",
        default="outputs/gpt2_gsm8k_distilled_lr2e5_acc4/model_final",
        help=(
            "Fine-tuned/distilled model name/path for the second run "
            "(default: outputs/gpt2_gsm8k_distilled_lr2e5_acc4/model_final)"
        ),
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated seeds. Each model runs once per seed.",
    )
    parser.add_argument(
        "--causal-build-cache-num-examples",
        type=int,
        default=200,
        help="Examples per corruption for cache build.",
    )
    parser.add_argument(
        "--causal-num-examples",
        type=int,
        default=100,
        help="Total examples for causal discovery/evaluation split.",
    )
    parser.add_argument(
        "--causal-num-edges",
        type=int,
        default=10,
        help="Number of edges evaluated in causal-eval.",
    )
    parser.add_argument(
        "--causal-node-types",
        default="res,mlp,att",
        help="Node types for causal candidate proposal and cache build.",
    )
    parser.add_argument(
        "--causal-bootstrap-samples",
        type=int,
        default=200,
        help="Bootstrap samples for causal-eval.",
    )
    parser.add_argument(
        "--causal-permutation-samples",
        type=int,
        default=200,
        help="Permutation samples for causal-eval.",
    )
    parser.add_argument(
        "--causal-build-cache-corruptions",
        default="name_swap,abba",
        help="Corruptions used for cache building.",
    )
    parser.add_argument(
        "--graph-builder",
        default="correlation_topk",
        help="Registered graph-builder strategy.",
    )
    parser.add_argument(
        "--causal-device",
        default=None,
        help="Optional device for causal-eval (cpu/cuda).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Forwarded to pipeline; continue if one stage fails.",
    )
    parser.add_argument(
        "--causal-overwrite",
        action="store_true",
        help="Allow overwriting non-empty causal output directories.",
    )
    parser.add_argument(
        "--pipeline-log-dir",
        default="outputs/pipeline_logs/publication",
        help="Directory for per-run pipeline logs.",
    )
    parser.add_argument(
        "--causal-output-root",
        default="outputs/causal_eval/publication",
        help="Root directory for causal-eval outputs.",
    )
    parser.add_argument(
        "--causal-cache-root",
        default=".cache/patch_effects/publication",
        help=(
            "Root directory for causal cache runs. "
            "Each model/seed run writes to a unique subdirectory."
        ),
    )
    parser.add_argument(
        "--uv-bin",
        default="uv",
        help="Executable name/path for uv (default: uv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=2,
        help=(
            "Number of model tracks to run in parallel. "
            "Seeds within each model always run sequentially to avoid cache conflicts. "
            "Safe values: 1 (default, sequential) or the number of models (one GPU per model)."
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        default="0,1",
        help=(
            "Comma-separated GPU IDs to assign to parallel tracks, e.g. '0,1,2,3'. "
            "Track i gets CUDA_VISIBLE_DEVICES=gpu_ids[i %% len(gpu_ids)]. "
            "Ignored when --num-parallel=1."
        ),
    )
    args = parser.parse_args()

    try:
        seeds = _parse_seed_csv(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))

    repo_root = _repo_root()
    run_stamp = _utc_stamp()
    models = [args.base_model, args.finetuned_model]
    total_runs = len(models) * len(seeds)

    print("=" * 72)
    print("PIG PUBLICATION COMPARISON RUN")
    print("=" * 72)
    print(f"Repository: {repo_root}")
    print(f"UTC run stamp: {run_stamp}")
    print(f"Models: {models[0]} -> {models[1]}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {total_runs}")
    print()

    gpu_ids: list[str] = []
    if args.gpu_ids:
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]

    def _gpu_env(track_index: int) -> dict | None:
        if not gpu_ids:
            return None
        return {"CUDA_VISIBLE_DEVICES": gpu_ids[track_index % len(gpu_ids)]}

    completed_count = 0
    completed_lock = threading.Lock()
    errors: list[str] = []

    def run_model_track(model_name: str, track_index: int) -> int:
        """Run all seeds for one model sequentially on its assigned GPU."""
        env = _gpu_env(track_index)
        gpu_label = f"GPU {env['CUDA_VISIBLE_DEVICES']}" if env else "default device"
        model_key = _model_key(model_name)
        for seed in seeds:
            run_key = f"{model_key}_{run_stamp}_seed{seed}"
            pipeline_log_path = (
                repo_root / args.pipeline_log_dir / f"pipeline_{run_key}.log"
            )
            causal_cache_dir = repo_root / args.causal_cache_root / model_key / run_key
            causal_output_dir = repo_root / args.causal_output_root / run_key
            pipeline_log_path.parent.mkdir(parents=True, exist_ok=True)
            causal_cache_dir.parent.mkdir(parents=True, exist_ok=True)
            causal_output_dir.parent.mkdir(parents=True, exist_ok=True)

            command = _build_pipeline_command(
                uv_bin=args.uv_bin,
                model_name=model_name,
                graph_builder=args.graph_builder,
                pipeline_log_path=pipeline_log_path,
                causal_cache_dir=causal_cache_dir,
                causal_output_dir=causal_output_dir,
                causal_seed=seed,
                causal_num_examples=args.causal_num_examples,
                causal_num_edges=args.causal_num_edges,
                causal_node_types=args.causal_node_types,
                causal_bootstrap_samples=args.causal_bootstrap_samples,
                causal_permutation_samples=args.causal_permutation_samples,
                causal_build_cache_num_examples=args.causal_build_cache_num_examples,
                causal_build_cache_corruptions=args.causal_build_cache_corruptions,
                causal_device=args.causal_device,
                continue_on_error=args.continue_on_error,
                causal_overwrite=args.causal_overwrite,
            )

            with completed_lock:
                nonlocal completed_count
                completed_count += 1
                run_num = completed_count

            _log(
                f"\n{'-' * 72}\n"
                f"Run {run_num}/{total_runs}  [{gpu_label}]\n"
                f"Model: {model_name}\n"
                f"Seed:  {seed}\n"
                f"Pipeline log: {pipeline_log_path}\n"
                f"Causal cache: {causal_cache_dir}\n"
                f"Causal output: {causal_output_dir}"
            )
            exit_code = _run_command(
                command, cwd=repo_root, dry_run=args.dry_run, env=env
            )
            if exit_code != 0:
                msg = f"[FAIL] Exit code {exit_code} for model={model_name} seed={seed}"
                _log(msg)
                errors.append(msg)
                return exit_code
            _log(f"[PASS] model={model_name} seed={seed}")
        return 0

    num_parallel = max(1, min(args.num_parallel, len(models)))
    if num_parallel == 1:
        for i, model_name in enumerate(models):
            rc = run_model_track(model_name, i)
            if rc != 0:
                return rc
    else:
        _log(f"Parallel mode: {num_parallel} model tracks running simultaneously.")
        if gpu_ids:
            for i, model_name in enumerate(models):
                _log(
                    f"  {model_name} → CUDA_VISIBLE_DEVICES={gpu_ids[i % len(gpu_ids)]}"
                )
        with ThreadPoolExecutor(max_workers=num_parallel) as pool:
            futures = {
                pool.submit(run_model_track, model_name, i): model_name
                for i, model_name in enumerate(models)
            }
            for future in as_completed(futures):
                model_name = futures[future]
                rc = future.result()
                if rc != 0 and not args.continue_on_error:
                    _log(
                        f"[ABORT] model={model_name} failed; stopping remaining tracks."
                    )
                    return rc

    if errors:
        _log(f"\n{len(errors)} track(s) failed:")
        for e in errors:
            _log(f"  {e}")
        return 1

    print("-" * 72)
    print(f"Completed {total_runs}/{total_runs} runs.")
    if args.dry_run:
        print("Dry run only: no commands executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
