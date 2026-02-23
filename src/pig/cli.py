"""Command-line interface for PIG."""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

from pig.patching import (
    build_axis_fingerprint,
    build_component_axis,
    build_model_fingerprint,
    default_patch_cache_dir,
)

VALIDATION_SEQUENCE = [
    "validate_story_1_1.py",
    "validate_story_1_2.py",
    "validate_story_2_1.py",
    "validate_story_2_2.py",
    "validate_story_3_1.py",
    "validate_story_4_1.py",
    "validate_story_4_2.py",
    "validate_story_5_1.py",
    "validate_story_5_2.py",
    "generate_pipeline_artifacts.py",
]

MODEL_OUTPUT_SUFFIXES = ("_gpt2", "_toy")
GRAPH_BUILDER_ENV_VAR = "PIG_GRAPH_BUILDER"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _scripts_dir() -> Path:
    return _repo_root() / "scripts"


def _model_output_suffix(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized in {"toy", "toy_model", "toy_transformer"}:
        return "_toy"
    return "_gpt2"


def _resolve_cache_dir(model_name: str, cache_dir: str | None) -> str:
    if cache_dir:
        return cache_dir
    return str(default_patch_cache_dir(model_name))


def _resolve_component_size_validation(
    requested_component_size: int | None,
    *,
    expected_component_axis_size: int,
    model_name: str,
    node_types: list[str],
) -> tuple[int, bool]:
    """Resolve component-size validation and reject incompatible explicit values."""
    if requested_component_size is None:
        return expected_component_axis_size, False
    if requested_component_size != expected_component_axis_size:
        node_types_label = ",".join(node_types) if node_types else "<none>"
        raise ValueError(
            "Incompatible --component-size value. "
            f"Received {requested_component_size}, but expected "
            f"{expected_component_axis_size} for model={model_name!r} "
            f"and node_types={node_types_label!r}. "
            "Remove --component-size to auto-derive it."
        )
    return requested_component_size, True


@contextlib.contextmanager
def _stream_stdio_to_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            yield log_file


def _write_log_line(log_file, message: str) -> None:
    if log_file is None:
        return
    log_file.write(f"{message.rstrip()}\n")
    log_file.flush()


def _snapshot_suffixed_outputs() -> dict[str, bytes]:
    outputs_dir = _repo_root() / "outputs"
    if not outputs_dir.exists():
        return {}

    snapshot: dict[str, bytes] = {}
    for path in outputs_dir.iterdir():
        if not path.is_file():
            continue
        if not path.stem.endswith(MODEL_OUTPUT_SUFFIXES):
            continue
        snapshot[path.name] = path.read_bytes()
    return snapshot


def _restore_suffixed_outputs(snapshot: dict[str, bytes]) -> int:
    if not snapshot:
        return 0

    outputs_dir = _repo_root() / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    restored_count = 0
    for file_name, content in snapshot.items():
        path = outputs_dir / file_name
        if path.exists():
            continue
        path.write_bytes(content)
        restored_count += 1

    return restored_count


def _suffix_output_files(model_name: str) -> int:
    outputs_dir = _repo_root() / "outputs"
    if not outputs_dir.exists():
        return 0

    suffix = _model_output_suffix(model_name)
    renamed_count = 0

    for path in outputs_dir.iterdir():
        if not path.is_file():
            continue
        if path.stem.endswith(MODEL_OUTPUT_SUFFIXES):
            continue

        candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
        if candidate.exists():
            duplicate_index = 2
            while True:
                candidate = path.with_name(
                    f"{path.stem}{suffix}_{duplicate_index}{path.suffix}"
                )
                if not candidate.exists():
                    break
                duplicate_index += 1

        path.rename(candidate)
        renamed_count += 1

    return renamed_count


def _run_script(
    script_name: str,
    model_name: str,
    graph_builder: str | None = None,
    log_file=None,
) -> int:
    script_path = _scripts_dir() / script_name
    if not script_path.exists():
        _write_log_line(log_file, f"[FAIL] Missing script: {script_path}")
        return 2

    env = {**os.environ, "PIG_MODEL_NAME": model_name}
    if graph_builder:
        env[GRAPH_BUILDER_ENV_VAR] = graph_builder

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(_repo_root()),
        env=env,
        stdout=log_file,
        stderr=log_file,
        check=False,
    )
    return result.returncode


def _run_causal_eval_subcommand(
    model_name: str,
    cache_dir: str | None = None,
    output_dir: str = "outputs/causal_eval",
    log_path: str | None = None,
    num_examples: int = 20,
    num_edges: int = 5,
    node_types: str = "att",
    component_size: int | None = None,
    allow_legacy_cache: bool = False,
    seed: int = 42,
    eps: float = 1e-6,
    bootstrap_samples: int = 200,
    permutation_samples: int = 200,
    device: str | None = None,
) -> int:
    resolved_cache_dir = _resolve_cache_dir(model_name, cache_dir)
    command = [
        sys.executable,
        "-m",
        "pig.cli",
        "causal-eval",
        "--model-name",
        model_name,
        "--cache-dir",
        resolved_cache_dir,
        "--output-dir",
        output_dir,
        "--num-examples",
        str(num_examples),
        "--num-edges",
        str(num_edges),
        "--node-types",
        node_types,
        "--seed",
        str(seed),
        "--eps",
        str(eps),
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--permutation-samples",
        str(permutation_samples),
    ]
    if component_size is not None:
        command.extend(["--component-size", str(component_size)])
    if allow_legacy_cache:
        command.append("--allow-legacy-cache")
    if log_path:
        command.extend(["--log-path", log_path])
    if device:
        command.extend(["--device", device])

    result = subprocess.run(
        command,
        cwd=str(_repo_root()),
        env={**os.environ},
        check=False,
    )
    return result.returncode


def _run_viewer_cache_subcommand(
    model_name: str,
    cache_dir: str | None = None,
    num_examples: int = 16,
    corruptions: str = "name_swap,abba",
    node_types: str = "att",
    seed: int = 42,
    log_file=None,
) -> int:
    resolved_cache_dir = _resolve_cache_dir(model_name, cache_dir)
    command = [
        sys.executable,
        "-m",
        "pig.web.cache_builder",
        "--model-name",
        model_name,
        "--cache-dir",
        resolved_cache_dir,
        "--num-examples",
        str(num_examples),
        "--corruptions",
        corruptions,
        "--node-types",
        node_types,
        "--seed",
        str(seed),
    ]
    result = subprocess.run(
        command,
        cwd=str(_repo_root()),
        env={**os.environ},
        stdout=log_file,
        stderr=log_file,
        check=False,
    )
    return result.returncode


def run_pipeline(
    model_name: str,
    stop_on_failure: bool = True,
    graph_builder: str | None = None,
    pipeline_log_path: str | None = None,
    with_causal_eval: bool = False,
    causal_cache_dir: str | None = None,
    causal_output_dir: str = "outputs/causal_eval",
    causal_log_path: str | None = None,
    causal_num_examples: int = 20,
    causal_num_edges: int = 5,
    causal_node_types: str = "att",
    causal_component_size: int | None = None,
    causal_allow_legacy_cache: bool = False,
    causal_seed: int = 42,
    causal_eps: float = 1e-6,
    causal_bootstrap_samples: int = 200,
    causal_permutation_samples: int = 200,
    causal_device: str | None = None,
    causal_build_cache: bool = False,
    causal_build_cache_num_examples: int = 200,
    causal_build_cache_corruptions: str = "name_swap,abba",
    causal_build_cache_node_types: str | None = None,
    causal_build_cache_seed: int | None = None,
) -> int:
    start_time = time.time()
    failures: list[tuple[str, int]] = []
    output_snapshot = _snapshot_suffixed_outputs()
    pipeline_log = (
        Path(pipeline_log_path)
        if pipeline_log_path
        else (_repo_root() / "outputs" / "pipeline.log")
    )
    pipeline_log.parent.mkdir(parents=True, exist_ok=True)

    completed_steps = 0
    total_steps = (
        len(VALIDATION_SEQUENCE)
        + (1 if with_causal_eval else 0)
        + (1 if with_causal_eval and causal_build_cache else 0)
    )
    progress = tqdm(
        total=total_steps,
        desc="pig pipeline",
        dynamic_ncols=True,
        file=sys.stderr,
    )

    with open(pipeline_log, "a", encoding="utf-8", buffering=1) as log_file:
        _write_log_line(log_file, "=" * 72)
        _write_log_line(
            log_file,
            f"Pipeline start model={model_name} graph_builder={graph_builder}",
        )
        halted = False
        for script_name in VALIDATION_SEQUENCE:
            progress.set_postfix_str(script_name)
            exit_code = _run_script(
                script_name,
                model_name,
                graph_builder=graph_builder,
                log_file=log_file,
            )
            completed_steps += 1
            progress.update(1)
            if exit_code != 0:
                failures.append((script_name, exit_code))
                _write_log_line(log_file, f"[FAIL] {script_name} exited with code {exit_code}")
                if stop_on_failure:
                    halted = True
                    break
            else:
                _write_log_line(log_file, f"[PASS] {script_name}")

        if halted and completed_steps < len(VALIDATION_SEQUENCE):
            progress.total = completed_steps + (1 if with_causal_eval else 0)
            progress.refresh()

        if with_causal_eval and causal_build_cache:
            progress.set_postfix_str("viewer-cache")
            cache_builder_code = _run_viewer_cache_subcommand(
                model_name=model_name,
                cache_dir=causal_cache_dir,
                num_examples=causal_build_cache_num_examples,
                corruptions=causal_build_cache_corruptions,
                node_types=causal_build_cache_node_types or causal_node_types,
                seed=causal_seed if causal_build_cache_seed is None else causal_build_cache_seed,
                log_file=log_file,
            )
            completed_steps += 1
            progress.update(1)
            if cache_builder_code != 0:
                failures.append(("viewer-cache", cache_builder_code))
                _write_log_line(
                    log_file,
                    f"[FAIL] viewer-cache exited with code {cache_builder_code}",
                )
                if stop_on_failure:
                    halted = True
            else:
                _write_log_line(log_file, "[PASS] viewer-cache")

        if with_causal_eval and (not halted or not stop_on_failure):
            progress.set_postfix_str("causal-eval")
            causal_code = _run_causal_eval_subcommand(
                model_name=model_name,
                cache_dir=causal_cache_dir,
                output_dir=causal_output_dir,
                log_path=causal_log_path,
                num_examples=causal_num_examples,
                num_edges=causal_num_edges,
                node_types=causal_node_types,
                component_size=causal_component_size,
                allow_legacy_cache=causal_allow_legacy_cache,
                seed=causal_seed,
                eps=causal_eps,
                bootstrap_samples=causal_bootstrap_samples,
                permutation_samples=causal_permutation_samples,
                device=causal_device,
            )
            completed_steps += 1
            progress.update(1)
            if causal_code != 0:
                failures.append(("causal-eval", causal_code))
                _write_log_line(log_file, f"[FAIL] causal-eval exited with code {causal_code}")
            else:
                _write_log_line(log_file, "[PASS] causal-eval")

        restored_count = _restore_suffixed_outputs(output_snapshot)
        if restored_count > 0:
            _write_log_line(log_file, f"Previously suffixed outputs restored: {restored_count}")

        elapsed = time.time() - start_time
        _write_log_line(log_file, f"Elapsed: {elapsed:.1f}s")

        if failures:
            _write_log_line(log_file, "Failures:")
            for script_name, code in failures:
                _write_log_line(log_file, f"- {script_name}: {code}")
            progress.close()
            return 1

        renamed_count = _suffix_output_files(model_name)
        if renamed_count > 0:
            _write_log_line(
                log_file, f"Output files renamed with model suffix: {renamed_count}"
            )

        _write_log_line(log_file, "All validation stories completed successfully.")

    progress.close()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pig", description="PIG CLI")
    subparsers = parser.add_subparsers(dest="command")

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run validate_story_1_1 .. validate_story_5_2 in order",
    )
    pipeline_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue executing remaining scripts even if one fails",
    )
    pipeline_parser.add_argument(
        "--model-name",
        default="gpt2",
        help="Model identifier to use (e.g. gpt2, toy_transformer)",
    )
    pipeline_parser.add_argument(
        "--graph-builder",
        default=None,
        help=(
            "Registered graph builder strategy "
            "(e.g. correlation_topk, abs_correlation_topk)"
        ),
    )
    pipeline_parser.add_argument(
        "--pipeline-log-path",
        default=None,
        help="Optional pipeline log file path (default: outputs/pipeline.log)",
    )
    pipeline_parser.add_argument(
        "--with-causal-eval",
        action="store_true",
        help="Run causal-eval after the standard pipeline scripts",
    )
    pipeline_parser.add_argument(
        "--causal-cache-dir",
        default=None,
        help=(
            "Cache directory used by causal-eval "
            "(default: .cache/patch_effects/<model_key>)"
        ),
    )
    pipeline_parser.add_argument(
        "--causal-output-dir",
        default="outputs/causal_eval",
        help="Output directory for causal-eval artifacts",
    )
    pipeline_parser.add_argument(
        "--causal-log-path",
        default=None,
        help="Optional log file path for causal-eval (default: <causal-output-dir>/causal_eval.log)",
    )
    pipeline_parser.add_argument(
        "--causal-num-examples",
        type=int,
        default=20,
        help=(
            "Total examples used for disjoint discovery/evaluation split in causal-eval"
        ),
    )
    pipeline_parser.add_argument(
        "--causal-num-edges",
        type=int,
        default=5,
        help="Number of edges to evaluate in causal-eval",
    )
    pipeline_parser.add_argument(
        "--causal-node-types",
        default="att",
        help="Node types for causal-eval candidate proposal",
    )
    pipeline_parser.add_argument(
        "--causal-component-size",
        type=int,
        default=None,
        help=(
            "Deprecated validation-only check for component-axis size. "
            "If omitted, it is auto-derived from --model-name and --causal-node-types. "
            "If provided and mismatched, causal-eval fails explicitly."
        ),
    )
    pipeline_parser.add_argument(
        "--causal-allow-legacy-cache",
        action="store_true",
        help=(
            "Allow legacy cache entries without modern metadata "
            "(disabled by default; may mix incompatible tensors)."
        ),
    )
    pipeline_parser.add_argument(
        "--causal-seed",
        type=int,
        default=42,
        help="Seed for causal-eval subset/stats",
    )
    pipeline_parser.add_argument(
        "--causal-eps",
        type=float,
        default=1e-6,
        help="Epsilon used by causal-eval metrics",
    )
    pipeline_parser.add_argument(
        "--causal-bootstrap-samples",
        type=int,
        default=200,
        help="Bootstrap samples for causal-eval",
    )
    pipeline_parser.add_argument(
        "--causal-permutation-samples",
        type=int,
        default=200,
        help="Permutation samples for causal-eval",
    )
    pipeline_parser.add_argument(
        "--causal-device",
        default=None,
        help="Device override for causal-eval model loading",
    )
    pipeline_parser.add_argument(
        "--causal-build-cache",
        action="store_true",
        help="Run viewer-cache before causal-eval to refresh causal cache",
    )
    pipeline_parser.add_argument(
        "--causal-build-cache-num-examples",
        type=int,
        default=200,
        help="Examples per corruption when rebuilding causal cache",
    )
    pipeline_parser.add_argument(
        "--causal-build-cache-corruptions",
        default="name_swap,abba",
        help="Corruptions used when rebuilding causal cache",
    )
    pipeline_parser.add_argument(
        "--causal-build-cache-node-types",
        default=None,
        help="Node types for cache rebuild (defaults to --causal-node-types)",
    )
    pipeline_parser.add_argument(
        "--causal-build-cache-seed",
        type=int,
        default=None,
        help="Seed for cache rebuild (defaults to --causal-seed)",
    )

    viewer_parser = subparsers.add_parser(
        "viewer",
        help="Start local websocket backend for interactive graph viewer",
    )
    viewer_parser.add_argument(
        "--cache-dir",
        default=".cache/patch_effects",
        help="Directory with cached patch-effect JSON files",
    )
    viewer_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for websocket server",
    )
    viewer_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for websocket server",
    )
    viewer_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k edges per node when building graphs",
    )
    viewer_parser.add_argument(
        "--max-edges",
        type=int,
        default=3000,
        help="Default edge cap per response",
    )

    cache_parser = subparsers.add_parser(
        "viewer-cache",
        help="Generate patch-effect cache files for local graph viewer",
    )
    cache_parser.add_argument(
        "--model-name",
        default="toy_transformer",
        help="Model identifier to use for cache generation",
    )
    cache_parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Output directory for patch-effect cache JSON files "
            "(default: .cache/patch_effects/<model_key>)"
        ),
    )
    cache_parser.add_argument(
        "--num-examples",
        type=int,
        default=16,
        help="Number of examples per corruption slice",
    )
    cache_parser.add_argument(
        "--corruptions",
        default="name_swap,abba",
        help="Comma-separated corruption names",
    )
    cache_parser.add_argument(
        "--node-types",
        default="res",
        help="Comma-separated node types (res,mlp,att)",
    )
    cache_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )

    causal_parser = subparsers.add_parser(
        "causal-eval",
        help="Run interventional causal A/B/C evaluation on top edge candidates",
    )
    causal_parser.add_argument(
        "--model-name",
        default="gpt2",
        help="Model identifier or local path (default: gpt2)",
    )
    causal_parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g. cpu, cuda). Default: auto",
    )
    causal_parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Directory with cached patch-effect JSON files "
            "(default: .cache/patch_effects/<model_key>)"
        ),
    )
    causal_parser.add_argument(
        "--output-dir",
        default="outputs/causal_eval",
        help="Directory for causal_eval.json/.npz and quicklook figure",
    )
    causal_parser.add_argument(
        "--log-path",
        default=None,
        help="Optional log file path (default: <output-dir>/causal_eval.log)",
    )
    causal_parser.add_argument(
        "--num-examples",
        type=int,
        default=20,
        help=(
            "Total examples used for disjoint discovery/evaluation split "
            "(evaluation runs on a strict subset)"
        ),
    )
    causal_parser.add_argument(
        "--num-edges",
        type=int,
        default=5,
        help="Number of directed candidate edges to evaluate",
    )
    causal_parser.add_argument(
        "--node-types",
        default="att",
        help="Comma-separated node types for candidate proposal (res,mlp,att)",
    )
    causal_parser.add_argument(
        "--component-size",
        type=int,
        default=None,
        help=(
            "Deprecated validation-only check for component-axis size. "
            "If omitted, it is auto-derived from --model-name and --node-types. "
            "If provided and mismatched, causal-eval fails explicitly."
        ),
    )
    causal_parser.add_argument(
        "--allow-legacy-cache",
        action="store_true",
        help=(
            "Allow legacy cache entries without modern metadata "
            "(disabled by default)."
        ),
    )
    causal_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset/candidates/bootstrap/permutation",
    )
    causal_parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Small epsilon for metric normalization",
    )
    causal_parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="Bootstrap resamples per metric",
    )
    causal_parser.add_argument(
        "--permutation-samples",
        type=int,
        default=200,
        help="Sign-flip permutation samples per metric",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in (None, "pipeline"):
        code = run_pipeline(
            model_name=getattr(args, "model_name", "gpt2"),
            stop_on_failure=not getattr(args, "continue_on_error", False),
            graph_builder=getattr(args, "graph_builder", None),
            pipeline_log_path=getattr(args, "pipeline_log_path", None),
            with_causal_eval=getattr(args, "with_causal_eval", False),
            causal_cache_dir=getattr(args, "causal_cache_dir", None),
            causal_output_dir=getattr(args, "causal_output_dir", "outputs/causal_eval"),
            causal_log_path=getattr(args, "causal_log_path", None),
            causal_num_examples=getattr(args, "causal_num_examples", 20),
            causal_num_edges=getattr(args, "causal_num_edges", 5),
            causal_node_types=getattr(args, "causal_node_types", "att"),
            causal_component_size=getattr(args, "causal_component_size", None),
            causal_allow_legacy_cache=getattr(
                args, "causal_allow_legacy_cache", False
            ),
            causal_seed=getattr(args, "causal_seed", 42),
            causal_eps=getattr(args, "causal_eps", 1e-6),
            causal_bootstrap_samples=getattr(args, "causal_bootstrap_samples", 200),
            causal_permutation_samples=getattr(
                args, "causal_permutation_samples", 200
            ),
            causal_device=getattr(args, "causal_device", None),
            causal_build_cache=getattr(args, "causal_build_cache", False),
            causal_build_cache_num_examples=getattr(
                args, "causal_build_cache_num_examples", 200
            ),
            causal_build_cache_corruptions=getattr(
                args, "causal_build_cache_corruptions", "name_swap,abba"
            ),
            causal_build_cache_node_types=getattr(
                args, "causal_build_cache_node_types", None
            ),
            causal_build_cache_seed=getattr(args, "causal_build_cache_seed", None),
        )
        raise SystemExit(code)

    if args.command == "viewer":
        viewer_module = "pig.web.api"
        viewer_command = [
            sys.executable,
            "-m",
            viewer_module,
            "--cache-dir",
            args.cache_dir,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--top-k",
            str(args.top_k),
            "--max-edges",
            str(args.max_edges),
        ]
        code = subprocess.run(
            viewer_command,
            cwd=str(_repo_root()),
            env={**os.environ},
            check=False,
        ).returncode
        raise SystemExit(code)

    if args.command == "viewer-cache":
        cache_module = "pig.web.cache_builder"
        cache_dir = _resolve_cache_dir(args.model_name, args.cache_dir)
        cache_command = [
            sys.executable,
            "-m",
            cache_module,
            "--model-name",
            args.model_name,
            "--cache-dir",
            cache_dir,
            "--num-examples",
            str(args.num_examples),
            "--corruptions",
            args.corruptions,
            "--node-types",
            args.node_types,
            "--seed",
            str(args.seed),
        ]
        code = subprocess.run(
            cache_command,
            cwd=str(_repo_root()),
            env={**os.environ},
            check=False,
        ).returncode
        raise SystemExit(code)

    if args.command == "causal-eval":
        from pig.causal import (
            CausalEvalConfig,
            evaluate_causal_edges,
            load_cached_patch_effect_tensors,
            propose_causal_edge_candidates,
            select_causal_subset_from_cache,
            split_discovery_evaluation_tensors,
        )
        from pig.model import create_model

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(args.log_path) if args.log_path else (output_dir / "causal_eval.log")
        with _stream_stdio_to_log(log_path) as log_file:
            node_types = [x.strip() for x in args.node_types.split(",") if x.strip()]
            cache_dir = _resolve_cache_dir(args.model_name, args.cache_dir)
            if args.allow_legacy_cache:
                print(
                    "[WARN] --allow-legacy-cache enabled: legacy tensors without "
                    "modern metadata may be reused."
                )
            print(
                "Starting causal-eval "
                f"(model={args.model_name}, examples={args.num_examples}, edges={args.num_edges})"
            )
            print(f"Log file: {log_path}")
            print(f"Cache dir: {cache_dir}")

            model = create_model(model_name=args.model_name, device=args.device)
            expected_component_axis = build_component_axis(
                node_types=node_types,
                num_heads=model.n_heads,
            )
            expected_component_axis_size = len(expected_component_axis)
            try:
                component_size_validation, uses_explicit_component_size = (
                    _resolve_component_size_validation(
                        args.component_size,
                        expected_component_axis_size=expected_component_axis_size,
                        model_name=args.model_name,
                        node_types=node_types,
                    )
                )
            except ValueError as exc:
                print(f"[FAIL] {exc}")
                raise SystemExit(2)
            if uses_explicit_component_size:
                print(
                    "[WARN] --component-size is deprecated and only used as "
                    "validation, not as primary cache filter."
                )
            else:
                print(
                    "[INFO] Auto-derived component-size validation from model+axis: "
                    f"{component_size_validation}"
                )
            expected_model_fingerprint = build_model_fingerprint(model)
            expected_axis_fingerprint = build_axis_fingerprint(expected_component_axis)

            tensors_result = load_cached_patch_effect_tensors(
                cache_dir=cache_dir,
                expected_model_name=getattr(model, "model_name", None),
                expected_model_fingerprint=expected_model_fingerprint,
                expected_axis_fingerprint=expected_axis_fingerprint,
                allow_legacy_cache=args.allow_legacy_cache,
                component_size=component_size_validation,
                return_stats=True,
            )
            tensors, cache_filter_stats = tensors_result
            eligible_tensors, subset_filter_stats = select_causal_subset_from_cache(
                tensors,
                num_examples=len(tensors),
                component_size=component_size_validation,
                seed=args.seed,
                require_clean_better=True,
                return_stats=True,
            )
            if not eligible_tensors:
                print(
                    "[FAIL] No eligible cached tensors found after strict cache filtering. "
                    "Regenerate cache for this model/eje with `pig viewer-cache`."
                )
                print(f"[INFO] Cache filter stats: {cache_filter_stats}")
                print(f"[INFO] Subset filter stats: {subset_filter_stats}")
                raise SystemExit(1)
            (
                discovery_subset,
                evaluation_subset,
                split_stats,
            ) = split_discovery_evaluation_tensors(
                eligible_tensors,
                max_total_examples=args.num_examples,
                seed=args.seed,
                discovery_fraction=0.5,
                return_stats=True,
            )
            if not discovery_subset or not evaluation_subset:
                print(
                    "[FAIL] Could not create disjoint discovery/evaluation subsets. "
                    "Increase --num-examples and/or rebuild cache with more examples."
                )
                print(f"[INFO] Discovery/evaluation split stats: {split_stats}")
                raise SystemExit(1)
            print(
                "[INFO] Causal split: "
                f"discovery={len(discovery_subset)} evaluation={len(evaluation_subset)} "
                f"from eligible={len(eligible_tensors)}"
            )

            candidates = propose_causal_edge_candidates(
                discovery_subset,
                num_edges=args.num_edges,
                node_types=tuple(node_types),
                enforce_direction=True,
                eps=args.eps,
            )
            if not candidates:
                print("[FAIL] No edge candidates were proposed")
                raise SystemExit(1)

            prompt_pairs = [tensor.prompt_pair for tensor in evaluation_subset]
            config = CausalEvalConfig(
                eps=args.eps,
                bootstrap_samples=args.bootstrap_samples,
                permutation_samples=args.permutation_samples,
                seed=args.seed,
            )

            result = evaluate_causal_edges(
                model=model,
                prompt_pairs=prompt_pairs,
                candidates=candidates,
                config=config,
                show_progress=True,
                progress_desc="causal-eval",
                progress_file=sys.__stderr__,
            )
            discovery_slice_counts = dict(
                sorted(
                    Counter(
                        str(t.prompt_pair.slice_label) for t in discovery_subset
                    ).items()
                )
            )
            evaluation_slice_counts = dict(
                sorted(
                    Counter(
                        str(t.prompt_pair.slice_label) for t in evaluation_subset
                    ).items()
                )
            )
            result.run_metadata.update(
                {
                    "cache_dir": cache_dir,
                    "output_dir": args.output_dir,
                    "log_path": str(log_path),
                    "component_size_validation": component_size_validation,
                    "component_size_validation_source": (
                        "explicit_cli"
                        if uses_explicit_component_size
                        else "derived_model_and_node_types"
                    ),
                    "component_size_validation_deprecated": uses_explicit_component_size,
                    "requested_component_size": args.component_size,
                    "allow_legacy_cache": args.allow_legacy_cache,
                    "expected_model_fingerprint": expected_model_fingerprint,
                    "expected_axis_fingerprint": expected_axis_fingerprint,
                    "expected_component_axis_size": expected_component_axis_size,
                    "cache_filter_stats": cache_filter_stats,
                    "subset_filter_stats": subset_filter_stats,
                    "discovery_evaluation_split": split_stats,
                    "eligible_tensor_count": len(eligible_tensors),
                    "discovery_tensor_count": len(discovery_subset),
                    "evaluation_tensor_count": len(evaluation_subset),
                    "discovery_slice_counts": discovery_slice_counts,
                    "evaluation_slice_counts": evaluation_slice_counts,
                    "requested_node_types": node_types,
                }
            )

            json_path = result.save_json(output_dir / "causal_eval.json")
            npz_path = result.save_npz(output_dir / "causal_eval.npz")
            print(f"Saved: {json_path}")
            print(f"Saved: {npz_path}")

            plot_script = _scripts_dir() / "plot_causal_eval.py"
            if plot_script.exists():
                quicklook_path = output_dir / "quicklook_causal.png"
                plot_code = subprocess.run(
                    [
                        sys.executable,
                        str(plot_script),
                        "--json-path",
                        str(json_path),
                        "--npz-path",
                        str(npz_path),
                        "--output-path",
                        str(quicklook_path),
                    ],
                    cwd=str(_repo_root()),
                    env={**os.environ},
                    stdout=log_file,
                    stderr=log_file,
                    check=False,
                ).returncode
                if plot_code == 0:
                    print(f"Saved: {quicklook_path}")
                else:
                    print(
                        "[WARN] quicklook plot script exited with "
                        f"code {plot_code}: {plot_script}"
                    )
            else:
                print(f"[WARN] Missing quicklook script: {plot_script}")

        raise SystemExit(0)

    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
