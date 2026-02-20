"""CLI tests for causal-eval command."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import pig.cli as cli
import pig.causal as causal
import pig.model as model_mod
from pig.causal import CausalEdgeCandidate, CausalEvalConfig, CausalEvalResult
from pig.prompts import PromptPair, SliceLabel


def test_cli_causal_eval_help(capsys):
    with pytest.raises(SystemExit) as exc:
        parser = cli._build_parser()
        parser.parse_args(["causal-eval", "--help"])
    assert exc.value.code == 0

    help_text = capsys.readouterr().out
    assert "causal-eval" in help_text
    assert "--num-examples" in help_text


def test_cli_causal_eval_short_run_with_mocks(monkeypatch, tmp_path):
    out_dir = tmp_path / "causal_out"
    prompt_pair = PromptPair(
        x_cln="John gave the book to Mary . Mary gave it back to",
        x_crp="Mary gave the book to Mary . Mary gave it back to",
        y_star="John",
        slice_label=SliceLabel(task="ioi", corruption="name_swap"),
        meta={},
    )

    def fake_load_cached_patch_effect_tensors(cache_dir):
        _ = cache_dir
        return [SimpleNamespace(prompt_pair=prompt_pair)]

    def fake_select_causal_subset_from_cache(
        tensors,
        num_examples,
        component_size,
        seed,
        require_clean_better,
    ):
        _ = (tensors, num_examples, component_size, seed, require_clean_better)
        return [SimpleNamespace(prompt_pair=prompt_pair)]

    def fake_propose_causal_edge_candidates(tensors, num_edges, node_types, enforce_direction, eps):
        _ = (tensors, num_edges, node_types, enforce_direction, eps)
        return [
            CausalEdgeCandidate(
                src_layer=0,
                src_token=0,
                src_node_type="att",
                src_head=0,
                dst_layer=1,
                dst_token=1,
                dst_node_type="att",
                dst_head=0,
            )
        ]

    def fake_create_model(model_name, device):
        _ = (model_name, device)
        return object()

    def fake_evaluate(model, prompt_pairs, candidates, config, **kwargs):
        _ = (model, prompt_pairs, candidates)
        _ = kwargs
        assert isinstance(config, CausalEvalConfig)
        arrays = {
            "edge_ids": np.asarray(["e1"], dtype=np.str_),
            "src_labels": np.asarray(["L0T0.att[0]"], dtype=np.str_),
            "dst_labels": np.asarray(["L1T1.att[0]"], dtype=np.str_),
            "I_mean": np.asarray([0.5]),
            "I_ci_low": np.asarray([0.4]),
            "I_ci_high": np.asarray([0.6]),
            "R_u_mean": np.asarray([0.4]),
            "R_v_mean": np.asarray([0.2]),
            "R_uv_mean": np.asarray([0.45]),
            "M_mean": np.asarray([0.25]),
            "R_u_clamp_v_mean": np.asarray([0.15]),
            "necessity_mean": np.asarray([0.25]),
            "necessity_ci_low": np.asarray([0.2]),
            "necessity_ci_high": np.asarray([0.3]),
        }
        return CausalEvalResult(
            model_name="gpt2",
            config=config,
            run_metadata={},
            edges=[{"edge_id": "e1"}],
            arrays=arrays,
        )

    monkeypatch.setattr(
        causal,
        "load_cached_patch_effect_tensors",
        fake_load_cached_patch_effect_tensors,
    )
    monkeypatch.setattr(
        causal,
        "select_causal_subset_from_cache",
        fake_select_causal_subset_from_cache,
    )
    monkeypatch.setattr(
        causal,
        "propose_causal_edge_candidates",
        fake_propose_causal_edge_candidates,
    )
    monkeypatch.setattr(causal, "evaluate_causal_edges", fake_evaluate)
    monkeypatch.setattr(model_mod, "create_model", fake_create_model)
    monkeypatch.setattr(cli, "_scripts_dir", lambda: tmp_path / "missing_scripts")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pig",
            "causal-eval",
            "--output-dir",
            str(out_dir),
            "--num-examples",
            "1",
            "--num-edges",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert (out_dir / "causal_eval.json").exists()
    assert (out_dir / "causal_eval.npz").exists()


def test_cli_pipeline_forwards_causal_flags(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pig",
            "pipeline",
            "--with-causal-eval",
            "--causal-build-cache",
            "--causal-num-examples",
            "9",
            "--causal-num-edges",
            "3",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert captured["with_causal_eval"] is True
    assert captured["causal_build_cache"] is True
    assert captured["causal_num_examples"] == 9
    assert captured["causal_num_edges"] == 3


def test_run_pipeline_executes_causal_eval_when_enabled(monkeypatch):
    calls: dict[str, int] = {"scripts": 0, "causal": 0}
    causal_kwargs: dict[str, object] = {}

    monkeypatch.setattr(cli, "_snapshot_suffixed_outputs", lambda: {})
    monkeypatch.setattr(cli, "_restore_suffixed_outputs", lambda snapshot: 0)
    monkeypatch.setattr(cli, "_suffix_output_files", lambda model_name: 0)

    def fake_run_script(script_name, model_name, graph_builder=None, log_file=None):
        _ = (script_name, model_name, graph_builder, log_file)
        calls["scripts"] += 1
        return 0

    def fake_run_causal_eval_subcommand(**kwargs):
        calls["causal"] += 1
        causal_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "_run_script", fake_run_script)
    monkeypatch.setattr(cli, "_run_causal_eval_subcommand", fake_run_causal_eval_subcommand)

    code = cli.run_pipeline(
        model_name="gpt2",
        with_causal_eval=True,
        causal_num_examples=9,
        causal_num_edges=3,
    )

    assert code == 0
    assert calls["scripts"] == len(cli.VALIDATION_SEQUENCE)
    assert calls["causal"] == 1
    assert causal_kwargs["num_examples"] == 9
    assert causal_kwargs["num_edges"] == 3


def test_run_pipeline_executes_cache_builder_before_causal_eval(monkeypatch):
    calls: dict[str, int] = {"scripts": 0, "cache": 0, "causal": 0}

    monkeypatch.setattr(cli, "_snapshot_suffixed_outputs", lambda: {})
    monkeypatch.setattr(cli, "_restore_suffixed_outputs", lambda snapshot: 0)
    monkeypatch.setattr(cli, "_suffix_output_files", lambda model_name: 0)

    def fake_run_script(script_name, model_name, graph_builder=None, log_file=None):
        _ = (script_name, model_name, graph_builder, log_file)
        calls["scripts"] += 1
        return 0

    def fake_run_viewer_cache_subcommand(**kwargs):
        _ = kwargs
        calls["cache"] += 1
        return 0

    def fake_run_causal_eval_subcommand(**kwargs):
        _ = kwargs
        calls["causal"] += 1
        return 0

    monkeypatch.setattr(cli, "_run_script", fake_run_script)
    monkeypatch.setattr(cli, "_run_viewer_cache_subcommand", fake_run_viewer_cache_subcommand)
    monkeypatch.setattr(cli, "_run_causal_eval_subcommand", fake_run_causal_eval_subcommand)

    code = cli.run_pipeline(
        model_name="gpt2",
        with_causal_eval=True,
        causal_build_cache=True,
    )

    assert code == 0
    assert calls["scripts"] == len(cli.VALIDATION_SEQUENCE)
    assert calls["cache"] == 1
    assert calls["causal"] == 1
