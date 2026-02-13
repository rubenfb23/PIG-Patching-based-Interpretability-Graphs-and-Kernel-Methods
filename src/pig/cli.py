"""Command-line interface for PIG."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

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
) -> int:
    script_path = _scripts_dir() / script_name
    if not script_path.exists():
        print(f"[FAIL] Missing script: {script_path}")
        return 2

    print(f"\n{'=' * 72}")
    print(f"Running {script_name}")
    print(f"{'=' * 72}")

    env = {**os.environ, "PIG_MODEL_NAME": model_name}
    if graph_builder:
        env[GRAPH_BUILDER_ENV_VAR] = graph_builder

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(_repo_root()),
        env=env,
        check=False,
    )
    return result.returncode


def run_pipeline(
    model_name: str,
    stop_on_failure: bool = True,
    graph_builder: str | None = None,
) -> int:
    start_time = time.time()
    failures: list[tuple[str, int]] = []
    output_snapshot = _snapshot_suffixed_outputs()

    print("PIG pipeline runner")
    print(f"Python: {sys.executable}")
    print(f"Repo: {_repo_root()}")
    print(f"Model: {model_name}")
    if graph_builder:
        print(f"Graph builder: {graph_builder}")

    for script_name in VALIDATION_SEQUENCE:
        exit_code = _run_script(
            script_name,
            model_name,
            graph_builder=graph_builder,
        )
        if exit_code != 0:
            failures.append((script_name, exit_code))
            print(f"[FAIL] {script_name} exited with code {exit_code}")
            if stop_on_failure:
                break
            continue
        print(f"[PASS] {script_name}")

    elapsed = time.time() - start_time
    print(f"\nElapsed: {elapsed:.1f}s")

    restored_count = _restore_suffixed_outputs(output_snapshot)
    if restored_count > 0:
        print(f"Previously suffixed outputs restored: {restored_count}")

    if failures:
        print("Failures:")
        for script_name, code in failures:
            print(f"- {script_name}: {code}")
        return 1

    renamed_count = _suffix_output_files(model_name)
    if renamed_count > 0:
        print("Output files renamed with model suffix: " f"{renamed_count}")

    print("All validation stories completed successfully.")
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

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in (None, "pipeline"):
        code = run_pipeline(
            model_name=args.model_name,
            stop_on_failure=not args.continue_on_error,
            graph_builder=args.graph_builder,
        )
        raise SystemExit(code)

    parser.print_help()
    raise SystemExit(2)
