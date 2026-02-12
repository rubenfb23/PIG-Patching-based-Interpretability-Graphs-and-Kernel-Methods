"""Command-line interface for PIG."""

from __future__ import annotations

import argparse
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
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _scripts_dir() -> Path:
    return _repo_root() / "scripts"


def _run_script(script_name: str) -> int:
    script_path = _scripts_dir() / script_name
    if not script_path.exists():
        print(f"[FAIL] Missing script: {script_path}")
        return 2

    print(f"\n{'=' * 72}")
    print(f"Running {script_name}")
    print(f"{'=' * 72}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(_repo_root()),
        check=False,
    )
    return result.returncode


def run_pipeline(stop_on_failure: bool = True) -> int:
    start_time = time.time()
    failures: list[tuple[str, int]] = []

    print("PIG pipeline runner")
    print(f"Python: {sys.executable}")
    print(f"Repo: {_repo_root()}")

    for script_name in VALIDATION_SEQUENCE:
        exit_code = _run_script(script_name)
        if exit_code != 0:
            failures.append((script_name, exit_code))
            print(f"[FAIL] {script_name} exited with code {exit_code}")
            if stop_on_failure:
                break
            continue
        print(f"[PASS] {script_name}")

    elapsed = time.time() - start_time
    print(f"\nElapsed: {elapsed:.1f}s")

    if failures:
        print("Failures:")
        for script_name, code in failures:
            print(f"- {script_name}: {code}")
        return 1

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

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in (None, "pipeline"):
        code = run_pipeline(stop_on_failure=not args.continue_on_error)
        raise SystemExit(code)

    parser.print_help()
    raise SystemExit(2)
