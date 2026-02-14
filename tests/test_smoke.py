"""Smoke test for CLMI pair pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from transformers import AutoConfig


def _pick_local_model() -> str | None:
    for model_name in ["gpt2", "sshleifer/tiny-gpt2"]:
        try:
            AutoConfig.from_pretrained(model_name, local_files_only=True)
            return model_name
        except OSError:
            continue
    return None


@pytest.mark.slow
def test_clmi_smoke_run_pair_with_cache(tmp_path: Path) -> None:
    model_name = _pick_local_model()
    if model_name is None:
        pytest.skip("No local GPT-2-compatible model is cached.")

    run_id = "pytest_smoke_clmi"
    summary_path = tmp_path / "smoke_summary.csv"

    base_cmd = [
        sys.executable,
        "scripts/run_pair.py",
        "--model",
        model_name,
        "--device",
        "cpu",
        "--local-files-only",
        "--seed",
        "0",
        "--pair-id",
        "0",
        "--overlap",
        "0.0",
        "--n-keys",
        "24",
        "--n-values",
        "24",
        "--k",
        "4",
        "--beta",
        "0.2",
        "--batch-size",
        "2",
        "--max-steps-a",
        "4",
        "--max-steps-b",
        "4",
        "--max-steps-a2",
        "4",
        "--eval-every",
        "2",
        "--run-id",
        run_id,
        "--summary-path",
        str(summary_path),
        "--smoke",
    ]

    first = subprocess.run(base_cmd, check=False, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr

    pair_metrics_path = Path("results") / "runs" / run_id / "pair_metrics.json"
    assert pair_metrics_path.exists()
    metrics_first = json.loads(pair_metrics_path.read_text())

    required_keys = [
        "AccA_MA",
        "AccA_MAB",
        "forgettingA",
        "NC_global",
        "mean_NC_layers",
        "KL_AB_BA",
    ]
    for key in required_keys:
        assert key in metrics_first

    second = subprocess.run(base_cmd, check=False, text=True, capture_output=True)
    assert second.returncode == 0, second.stderr

    metrics_second = json.loads(pair_metrics_path.read_text())
    assert metrics_second["cache_hit_A"] >= metrics_first["cache_hit_A"]
    assert metrics_second["cache_hit_B"] >= metrics_first["cache_hit_B"]

    assert (Path("results") / "runs" / run_id / "nc_layers.csv").exists()
    assert (Path("results") / "runs" / run_id / "hysteresis_curve.csv").exists()
