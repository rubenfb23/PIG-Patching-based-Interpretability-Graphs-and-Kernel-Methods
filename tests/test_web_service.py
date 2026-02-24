"""Unit tests for pig.web.service."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pig.patching import ComponentSpec, PatchEffectDataset, PatchEffectTensor
from pig.prompts import PromptPair, SliceLabel
from pig.web.schemas import ViewFilter
from pig.web.service import GraphViewerService, ViewerConfig, load_dataset_from_cache


def _make_tensor(index: int, slice_label: SliceLabel) -> PatchEffectTensor:
    effects = np.array(
        [
            [[0.2 + index], [0.1 + index], [0.0 + index]],
            [[-0.1 + index], [0.3 + index], [0.05 + index]],
        ],
        dtype=np.float32,
    )
    pair = PromptPair(
        x_cln=f"clean-{index}",
        x_crp=f"corrupt-{index}",
        y_star="target",
        slice_label=slice_label,
        meta={},
    )
    return PatchEffectTensor(
        effects=effects,
        component_axis=[ComponentSpec(node_type="res")],
        prompt_pair=pair,
        base_score=-1.0,
        clean_score=1.0,
    )


def _make_dataset() -> PatchEffectDataset:
    dataset = PatchEffectDataset()
    for corruption in ("name_swap", "abba"):
        slice_label = SliceLabel(task="ioi", corruption=corruption)
        for index in range(4):
            dataset.add(_make_tensor(index=index, slice_label=slice_label))
    return dataset


def test_initial_payload_includes_first_graph() -> None:
    service = GraphViewerService(_make_dataset(), ViewerConfig(top_k=2))

    payload = service.initial_payload()

    assert payload["slices"]
    assert payload["graph"]["nodes"]
    assert payload["graph"]["stats"]["selected_nodes"] > 0


def test_filtered_graph_applies_layer_token_and_edge_limits() -> None:
    service = GraphViewerService(
        _make_dataset(), ViewerConfig(top_k=3, max_edges_default=32)
    )
    payload = service.filtered_graph(
        ViewFilter(
            slice_id="ioi/name_swap",
            layer_min=0,
            layer_max=0,
            token_min=0,
            token_max=1,
            min_abs_weight=0.0,
            max_edges=2,
        )
    )

    assert payload["stats"]["selected_nodes"] == 2
    assert payload["stats"]["selected_edges"] <= 2
    assert all(node["layer"] == 0 for node in payload["nodes"])
    assert all(0 <= node["token"] <= 1 for node in payload["nodes"])


def test_filtered_graph_rejects_unknown_slice() -> None:
    service = GraphViewerService(_make_dataset(), ViewerConfig())

    with pytest.raises(ValueError, match="Unknown slice"):
        service.filtered_graph(ViewFilter(slice_id="missing/slice"))


def test_load_dataset_from_cache_reads_json_files(tmp_path: Path) -> None:
    tensor = _make_tensor(
        index=0, slice_label=SliceLabel(task="ioi", corruption="abba")
    )
    cache_path = tmp_path / "sample.json"
    cache_path.write_text(json.dumps(tensor.to_dict()), encoding="utf-8")

    dataset = load_dataset_from_cache(tmp_path)

    assert len(dataset) == 1
    loaded = dataset[0]
    assert loaded.shape == tensor.shape
    assert loaded.prompt_pair.slice_label.corruption == "abba"


def test_load_dataset_from_cache_requires_existing_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No cache files found"):
        load_dataset_from_cache(tmp_path)


def test_load_dataset_from_cache_selects_consistent_axis_group(
    tmp_path: Path,
) -> None:
    slice_label = SliceLabel(task="ioi", corruption="name_swap")

    res_tensor = _make_tensor(index=0, slice_label=slice_label)

    rich_pair = PromptPair(
        x_cln="clean-rich",
        x_crp="corrupt-rich",
        y_star="target",
        slice_label=slice_label,
        meta={},
    )
    rich_tensor = PatchEffectTensor(
        effects=np.ones((2, 3, 2), dtype=np.float32),
        component_axis=[
            ComponentSpec(node_type="mlp"),
            ComponentSpec(node_type="res"),
        ],
        prompt_pair=rich_pair,
        base_score=-1.0,
        clean_score=1.0,
    )

    (tmp_path / "res.json").write_text(
        json.dumps(res_tensor.to_dict()), encoding="utf-8"
    )
    (tmp_path / "rich_a.json").write_text(
        json.dumps(rich_tensor.to_dict()), encoding="utf-8"
    )
    (tmp_path / "rich_b.json").write_text(
        json.dumps(rich_tensor.to_dict()), encoding="utf-8"
    )

    dataset = load_dataset_from_cache(tmp_path)

    assert len(dataset) == 2
    assert dataset[0].num_components == 2
    assert dataset[1].num_components == 2


def test_load_dataset_from_cache_auto_selects_latest_run_subdirectory(
    tmp_path: Path,
) -> None:
    old_run = tmp_path / "run_old"
    new_run = tmp_path / "run_new"
    old_run.mkdir()
    new_run.mkdir()

    old_tensor = _make_tensor(
        index=0, slice_label=SliceLabel(task="ioi", corruption="abba")
    )
    new_tensor = _make_tensor(
        index=1, slice_label=SliceLabel(task="ioi", corruption="name_swap")
    )

    old_file = old_run / "tensor.json"
    new_file = new_run / "tensor.json"
    old_file.write_text(json.dumps(old_tensor.to_dict()), encoding="utf-8")
    new_file.write_text(json.dumps(new_tensor.to_dict()), encoding="utf-8")

    os.utime(old_file, ns=(1, 1))
    os.utime(new_file, ns=(2, 2))

    dataset = load_dataset_from_cache(tmp_path)

    assert len(dataset) == 1
    loaded = dataset[0]
    assert loaded.prompt_pair.x_cln == "clean-1"
    assert loaded.prompt_pair.slice_label.corruption == "name_swap"
