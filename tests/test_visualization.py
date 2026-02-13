"""Tests for visualization utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pig.patching import ComponentSpec, PatchEffectDataset, PatchEffectTensor
from pig.prompts import PromptPair, SliceLabel
from pig.visualization import (
    plot_patching_heatmap_layer_token,
    prepare_patching_heatmap_inputs,
)


def _build_nodes_table(
    n_layers: int,
    n_tokens: int,
    components: list[str],
) -> dict[str, np.ndarray]:
    node_index: list[int] = []
    layer: list[int] = []
    token_pos: list[int] = []
    component: list[str] = []

    cursor = 0
    for layer_idx in range(n_layers):
        for token_idx in range(n_tokens):
            for comp in components:
                node_index.append(cursor)
                layer.append(layer_idx)
                token_pos.append(token_idx)
                component.append(comp)
                cursor += 1

    return {
        "node_index": np.asarray(node_index, dtype=np.int64),
        "layer": np.asarray(layer, dtype=np.int64),
        "token_pos": np.asarray(token_pos, dtype=np.int64),
        "component": np.asarray(component, dtype=object),
    }


def test_plot_patching_heatmap_layer_token_dummy_acceptance(tmp_path: Path):
    """The heatmap matrix has expected shape and diverging center at 0."""
    n_layers = 3
    n_tokens = 4
    components = ["res", "mlp"]
    nodes_df = _build_nodes_table(n_layers, n_tokens, components)

    n_examples = 5
    n_nodes = len(nodes_df["node_index"])
    E = np.zeros((n_examples, n_nodes), dtype=np.float32)

    # Fill residual component with structured values and keep MLP very large to
    # ensure component filtering is respected.
    for row, (layer_idx, token_idx, comp) in enumerate(
        zip(nodes_df["layer"], nodes_df["token_pos"], nodes_df["component"])
    ):
        if comp == "res":
            E[0, row] = float(layer_idx - token_idx)
            E[1, row] = float(0.5 * (layer_idx - token_idx))
            E[2, row] = float(10.0)  # filtered out by subset
            E[3, row] = float(-10.0)  # filtered out by subset
            E[4, row] = float(5.0)  # filtered out by slice
        else:
            E[:, row] = 100.0

    examples_df = {
        "slice_id": np.asarray(
            [
                "ioi:name_swap",
                "ioi:name_swap",
                "ioi:name_swap",
                "ioi:name_swap",
                "ioi:abba",
            ],
            dtype=object,
        ),
        "clean_correct": np.asarray([True, True, True, False, True], dtype=bool),
        "corrupted_correct": np.asarray([False, False, True, False, False], dtype=bool),
    }

    output_png = tmp_path / "A_heatmap_layer_token__ioi-name-swap.png"
    output_html = tmp_path / "A_heatmap_layer_token__ioi-name-swap.html"
    output_json = tmp_path / "A_heatmap_layer_token__ioi-name-swap.json"

    result = plot_patching_heatmap_layer_token(
        E=E,
        nodes_df=nodes_df,
        examples_df=examples_df,
        slice_filter="ioi:name_swap",
        component="resid",
        agg="mean",
        subset="clean_correct_corrupted_wrong",
        add_contours=True,
        output_png=output_png,
        output_html=output_html,
        output_json=output_json,
    )

    matrix = result["matrix"]
    assert matrix.shape == (n_layers, n_tokens)

    # Two examples selected by subset, both using residual-only values.
    assert matrix[2, 0] == pytest.approx(1.5)
    assert matrix[0, 3] == pytest.approx(-2.25)

    metadata = result["metadata"]
    assert metadata["center"] == 0.0
    assert metadata["vmin"] == pytest.approx(-metadata["vmax"])
    assert metadata["num_examples_selected"] == 2

    assert output_png.exists()
    assert output_html.exists()
    assert output_json.exists()

    saved_metadata = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved_metadata["matrix_shape"] == [n_layers, n_tokens]
    assert saved_metadata["center"] == 0.0


def test_prepare_patching_heatmap_inputs_shapes() -> None:
    """PatchEffectDataset conversion yields valid E/nodes/examples tables."""
    dataset = PatchEffectDataset()

    for idx, corruption in enumerate(["name_swap", "abba"]):
        pair = PromptPair(
            x_cln=f"clean {idx}",
            x_crp=f"corrupt {idx}",
            y_star="target",
            slice_label=SliceLabel(task="ioi", corruption=corruption),
            meta={},
        )
        effects = np.full((2, 3, 1), fill_value=idx + 1, dtype=np.float32)
        dataset.add(
            PatchEffectTensor(
                effects=effects,
                component_axis=[ComponentSpec(node_type="res")],
                prompt_pair=pair,
                base_score=-2.0,
                clean_score=-1.0,
            )
        )

    E, nodes_df, examples_df = prepare_patching_heatmap_inputs(dataset)

    assert E.shape == (2, 2 * 3 * 1)
    assert nodes_df["layer"].shape[0] == E.shape[1]
    assert examples_df["slice_id"].shape[0] == E.shape[0]
    assert set(examples_df["slice_id"].tolist()) == {"ioi:name_swap", "ioi:abba"}
