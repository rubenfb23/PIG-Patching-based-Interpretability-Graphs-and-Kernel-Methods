"""CLI tests for causal-eval command."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import pig.cli as cli
import pig.causal as causal
import pig.model as model_mod
from pig.causal import CausalEdgeCandidate, CausalEvalConfig, CausalEvalResult
from pig.patching import (
    PATCH_CACHE_SCHEMA_VERSION,
    ComponentSpec,
    PatchEffectTensor,
    build_axis_fingerprint,
)
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
    prompt_pair_discovery = PromptPair(
        x_cln="John gave the book to Mary . Mary gave it back to",
        x_crp="Mary gave the book to Mary . Mary gave it back to",
        y_star="John",
        slice_label=SliceLabel(task="ioi", corruption="name_swap"),
        meta={},
    )
    prompt_pair_evaluation = PromptPair(
        x_cln="Alice gave the book to Bob . Bob gave it back to",
        x_crp="Bob gave the book to Bob . Bob gave it back to",
        y_star="Alice",
        slice_label=SliceLabel(task="ioi", corruption="abba"),
        meta={},
    )

    def fake_load_cached_patch_effect_tensors(**kwargs):
        _ = kwargs
        return (
            [
                SimpleNamespace(
                    prompt_pair=prompt_pair_discovery,
                    clean_score=1.0,
                    base_score=0.0,
                ),
                SimpleNamespace(
                    prompt_pair=prompt_pair_evaluation,
                    clean_score=1.0,
                    base_score=0.0,
                ),
            ],
            {},
        )

    def fake_select_causal_subset_from_cache(
        tensors,
        num_examples,
        component_size,
        seed,
        require_clean_better,
        return_stats,
    ):
        _ = (
            tensors,
            num_examples,
            component_size,
            seed,
            require_clean_better,
            return_stats,
        )
        return list(tensors), {"selected_count": len(tensors)}

    def fake_split_discovery_evaluation_tensors(
        tensors,
        max_total_examples,
        seed,
        discovery_fraction,
        return_stats,
    ):
        _ = (max_total_examples, seed, discovery_fraction, return_stats)
        return (
            [tensors[0]],
            [tensors[1]],
            {"selected_discovery_count": 1, "selected_evaluation_count": 1},
        )

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
        return SimpleNamespace(n_heads=4, model_name="toy_transformer")

    def fake_evaluate(model, prompt_pairs, candidates, config, **kwargs):
        _ = (model, prompt_pairs, candidates)
        _ = kwargs
        assert isinstance(config, CausalEvalConfig)
        assert len(prompt_pairs) == 1
        assert prompt_pairs[0].x_cln == prompt_pair_evaluation.x_cln
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
        "split_discovery_evaluation_tensors",
        fake_split_discovery_evaluation_tensors,
    )
    monkeypatch.setattr(
        causal,
        "propose_causal_edge_candidates",
        fake_propose_causal_edge_candidates,
    )
    monkeypatch.setattr(causal, "evaluate_causal_edges", fake_evaluate)
    monkeypatch.setattr(model_mod, "create_model", fake_create_model)
    monkeypatch.setattr(cli, "build_model_fingerprint", lambda model: "toy_fp")
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
            "2",
            "--num-edges",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert (out_dir / "causal_eval.json").exists()
    assert (out_dir / "causal_eval.npz").exists()


def test_cli_causal_eval_ignores_other_model_tensors_in_shared_cache(
    monkeypatch, tmp_path
):
    shared_cache = tmp_path / "shared_cache"
    shared_cache.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "causal_out"
    axis_4 = [ComponentSpec(node_type="att", head=head) for head in range(4)]

    toy_prompt = PromptPair(
        x_cln="toy-clean-1",
        x_crp="toy-corrupt-1",
        y_star="target",
        slice_label=SliceLabel(task="ioi", corruption="name_swap"),
        meta={},
    )
    toy_prompt_2 = PromptPair(
        x_cln="toy-clean-2",
        x_crp="toy-corrupt-2",
        y_star="target",
        slice_label=SliceLabel(task="ioi", corruption="abba"),
        meta={},
    )
    gpt_prompt = PromptPair(
        x_cln="gpt-clean",
        x_crp="gpt-corrupt",
        y_star="target",
        slice_label=SliceLabel(task="ioi", corruption="name_swap"),
        meta={},
    )

    def make_tensor(prompt_pair: PromptPair, model_name: str, model_fingerprint: str):
        return PatchEffectTensor(
            effects=np.ones((2, 3, 4), dtype=np.float32),
            component_axis=axis_4,
            prompt_pair=prompt_pair,
            base_score=-1.0,
            clean_score=1.0,
            cache_schema_version=PATCH_CACHE_SCHEMA_VERSION,
            model_name=model_name,
            model_fingerprint=model_fingerprint,
            axis_fingerprint=build_axis_fingerprint(axis_4),
            node_types=["att"],
            created_at_utc="2026-02-23T00:00:00+00:00",
            is_legacy_cache_entry=False,
        )

    toy_tensor = make_tensor(toy_prompt, "toy_transformer", "toy_fp")
    toy_tensor_2 = make_tensor(toy_prompt_2, "toy_transformer", "toy_fp")
    gpt_tensor = make_tensor(gpt_prompt, "gpt2", "gpt2_fp")
    (shared_cache / "toy.json").write_text(json.dumps(toy_tensor.to_dict()), encoding="utf-8")
    (shared_cache / "toy_2.json").write_text(
        json.dumps(toy_tensor_2.to_dict()), encoding="utf-8"
    )
    (shared_cache / "gpt.json").write_text(json.dumps(gpt_tensor.to_dict()), encoding="utf-8")

    def fake_create_model(model_name, device):
        _ = (model_name, device)
        return SimpleNamespace(n_heads=4, model_name="toy_transformer")

    captured_prompt_pairs: dict[str, list[PromptPair]] = {}
    captured_discovery_pairs: dict[str, list[PromptPair]] = {}

    def fake_propose_causal_edge_candidates(
        tensors, num_edges, node_types, enforce_direction, eps
    ):
        _ = (num_edges, node_types, enforce_direction, eps)
        captured_discovery_pairs["pairs"] = [tensor.prompt_pair for tensor in tensors]
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

    def fake_evaluate(model, prompt_pairs, candidates, config, **kwargs):
        _ = (model, candidates, kwargs)
        captured_prompt_pairs["pairs"] = list(prompt_pairs)
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
            model_name="toy_transformer",
            config=config,
            run_metadata={},
            edges=[{"edge_id": "e1"}],
            arrays=arrays,
        )

    monkeypatch.setattr(model_mod, "create_model", fake_create_model)
    monkeypatch.setattr(causal, "propose_causal_edge_candidates", fake_propose_causal_edge_candidates)
    monkeypatch.setattr(causal, "evaluate_causal_edges", fake_evaluate)
    monkeypatch.setattr(cli, "build_model_fingerprint", lambda model: "toy_fp")
    monkeypatch.setattr(cli, "_scripts_dir", lambda: tmp_path / "missing_scripts")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pig",
            "causal-eval",
            "--model-name",
            "toy_transformer",
            "--cache-dir",
            str(shared_cache),
            "--output-dir",
            str(out_dir),
            "--num-examples",
            "4",
            "--num-edges",
            "1",
            "--node-types",
            "att",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert len(captured_prompt_pairs["pairs"]) == 1
    assert len(captured_discovery_pairs["pairs"]) == 1
    assert captured_prompt_pairs["pairs"][0].x_cln != gpt_prompt.x_cln
    assert captured_discovery_pairs["pairs"][0].x_cln != gpt_prompt.x_cln
    assert captured_prompt_pairs["pairs"][0].x_cln != captured_discovery_pairs["pairs"][0].x_cln

    run_payload = json.loads((out_dir / "causal_eval.json").read_text(encoding="utf-8"))
    run_metadata = run_payload["run_metadata"]
    reasons = run_payload["run_metadata"]["cache_filter_stats"]["discard_reasons"]
    assert reasons["model_name_mismatch"] == 1
    assert run_metadata["discovery_tensor_count"] == 1
    assert run_metadata["evaluation_tensor_count"] == 1
    assert run_metadata["discovery_evaluation_split"]["selected_discovery_count"] == 1
    assert run_metadata["discovery_evaluation_split"]["selected_evaluation_count"] == 1


def test_cli_causal_eval_fails_on_component_size_mismatch(monkeypatch, tmp_path):
    out_dir = tmp_path / "causal_out"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    load_called = {"value": False}

    def fake_create_model(model_name, device):
        _ = (model_name, device)
        return SimpleNamespace(n_heads=4, model_name="toy_transformer")

    def fake_load_cached_patch_effect_tensors(**kwargs):
        _ = kwargs
        load_called["value"] = True
        return ([], {})

    monkeypatch.setattr(model_mod, "create_model", fake_create_model)
    monkeypatch.setattr(causal, "load_cached_patch_effect_tensors", fake_load_cached_patch_effect_tensors)
    monkeypatch.setattr(cli, "_scripts_dir", lambda: tmp_path / "missing_scripts")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pig",
            "causal-eval",
            "--model-name",
            "toy_transformer",
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(out_dir),
            "--component-size",
            "14",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    assert load_called["value"] is False
    log_text = (out_dir / "causal_eval.log").read_text(encoding="utf-8")
    assert "Incompatible --component-size value" in log_text
    assert "expected 4" in log_text


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
