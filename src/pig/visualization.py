"""Visualization utilities for pipeline artifacts.

This module generates the output figures expected by the project, including
slice-level patching heatmaps and embedding projections.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from pig.embeddings import WLFeatureMatrix
from pig.patching import NODE_TYPE_RES, PatchEffectDataset
from pig.prompts import SliceLabel

_COMPONENT_ALIASES = {
    "res": "res",
    "resid": "res",
    "residual": "res",
    "mlp": "mlp",
    "att": "att",
    "attention": "att",
}

_SLICE_FILTER_COLUMN = "slice_id"
_DEFAULT_SUBSET = "all"


def _normalize_token_label(token: str, max_len: int = 16) -> str:
    compact = token.replace("\n", "\\n").strip()
    if compact == "":
        compact = "<space>"
    if len(compact) > max_len:
        return f"{compact[: max_len - 1]}…"
    return compact


def _build_tick_labels(tokens: list[str]) -> list[str]:
    labels: list[str] = []
    for index, token in enumerate(tokens, start=1):
        normalized = _normalize_token_label(token)
        labels.append(f"t{index} ({normalized})")
    return labels


def _get_residual_component_index(dataset: PatchEffectDataset) -> int:
    component_axis = dataset.get_component_axis()
    if not component_axis:
        return 0

    for index, component in enumerate(component_axis):
        if component.node_type == NODE_TYPE_RES:
            return index

    return 0


def _table_column(table: Any, column_name: str) -> NDArray[np.object_]:
    try:
        column = table[column_name]
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Missing required column: '{column_name}'") from exc

    values = np.asarray(column)
    if values.ndim != 1:
        values = values.reshape(-1)
    return values


def _find_first_column(table: Any, candidates: Sequence[str]) -> str:
    for column in candidates:
        try:
            _table_column(table, column)
            return column
        except ValueError:
            continue
    raise ValueError(f"None of the required columns exist: {list(candidates)}")


def _sanitize_slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    normalized = normalized.strip("-")
    return normalized or "all"


def _normalize_component(component: str) -> str:
    normalized = component.strip().lower()
    return _COMPONENT_ALIASES.get(normalized, normalized)


def _resolve_slice_filter(slice_filter: str | Sequence[str] | None) -> list[str]:
    if slice_filter is None:
        return []
    if isinstance(slice_filter, str):
        return [slice_filter]
    return [str(item) for item in slice_filter]


def _resolve_subset_mask(examples_df: Any, subset: str) -> NDArray[np.bool_]:
    if subset in {"", "all"}:
        return np.ones_like(_table_column(examples_df, _SLICE_FILTER_COLUMN), dtype=bool)

    normalized = subset.strip().lower()
    accepted = {
        "clean_correct_corrupted_wrong",
        "clean_correct_and_corrupted_wrong",
        "clean_success_corrupted_failure",
    }
    if normalized not in accepted:
        raise ValueError(
            "Unknown subset. Supported values: all, "
            "clean_correct_corrupted_wrong"
        )

    clean_correct = _table_column(examples_df, "clean_correct").astype(bool)
    corrupted_correct = _table_column(examples_df, "corrupted_correct").astype(bool)
    return clean_correct & ~corrupted_correct


def _apply_example_filters(
    base_mask: NDArray[np.bool_],
    examples_df: Any,
    example_filters: Optional[dict[str, Any]],
) -> NDArray[np.bool_]:
    if not example_filters:
        return base_mask

    filtered = base_mask.copy()
    for column_name, expected in example_filters.items():
        column_values = _table_column(examples_df, column_name)
        if callable(expected):
            column_mask = np.asarray(expected(column_values), dtype=bool)
        elif isinstance(expected, (list, tuple, set, np.ndarray)):
            column_mask = np.isin(column_values, list(expected))
        else:
            column_mask = column_values == expected

        if column_mask.shape != filtered.shape:
            raise ValueError(
                f"Filter on '{column_name}' returned mismatched shape: "
                f"{column_mask.shape} != {filtered.shape}"
            )
        filtered &= column_mask

    return filtered


def _resolve_node_indices(nodes_df: Any, n_nodes: int) -> NDArray[np.int64]:
    try:
        node_indices = _table_column(nodes_df, "node_index").astype(np.int64)
    except ValueError:
        node_indices = np.arange(n_nodes, dtype=np.int64)

    if len(node_indices) != n_nodes:
        raise ValueError(
            "nodes_df row count must match E.shape[1] or include a full "
            "node_index mapping"
        )
    if np.unique(node_indices).size != node_indices.size:
        raise ValueError("node_index values must be unique")
    if int(node_indices.min()) < 0 or int(node_indices.max()) >= n_nodes:
        raise ValueError("node_index values must map to valid E columns")

    return node_indices


def prepare_patching_heatmap_inputs(
    dataset: PatchEffectDataset,
) -> tuple[NDArray[np.float32], dict[str, NDArray[Any]], dict[str, NDArray[Any]]]:
    """Convert ``PatchEffectDataset`` into ``(E, nodes_df, examples_df)`` tables.

    The returned ``nodes_df`` and ``examples_df`` are dict-like tabular objects
    compatible with :func:`plot_patching_heatmap_layer_token`.
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    component_axis = dataset.get_component_axis()
    component_labels = [spec.node_type for spec in component_axis] or [NODE_TYPE_RES]

    num_layers = dataset[0].num_layers
    min_tokens = min(tensor.num_tokens for tensor in dataset)
    num_components = len(component_labels)
    num_nodes = num_layers * min_tokens * num_components

    E = np.zeros((len(dataset), num_nodes), dtype=np.float32)

    example_ids: list[int] = []
    slice_ids: list[str] = []
    tasks: list[str] = []
    corruptions: list[str] = []
    clean_scores: list[float] = []
    corrupted_scores: list[float] = []
    clean_better_than_corrupted: list[bool] = []

    for example_index, tensor in enumerate(dataset):
        if tensor.num_layers != num_layers:
            raise ValueError("Inconsistent layer count across patching dataset")
        if tensor.num_components != num_components:
            raise ValueError("Inconsistent component axis across patching dataset")

        flattened = tensor.effects[:, :min_tokens, :].reshape(-1)
        E[example_index] = flattened

        slice_label = str(tensor.prompt_pair.slice_label)
        example_ids.append(example_index)
        slice_ids.append(slice_label)
        tasks.append(tensor.prompt_pair.slice_label.task)
        corruptions.append(tensor.prompt_pair.slice_label.corruption)
        clean_scores.append(float(tensor.clean_score))
        corrupted_scores.append(float(tensor.base_score))
        clean_better_than_corrupted.append(tensor.clean_score > tensor.base_score)

    node_index: list[int] = []
    layer: list[int] = []
    token_pos: list[int] = []
    component: list[str] = []
    component_index: list[int] = []

    cursor = 0
    for layer_idx in range(num_layers):
        for token_idx in range(min_tokens):
            for component_idx, component_name in enumerate(component_labels):
                node_index.append(cursor)
                layer.append(layer_idx)
                token_pos.append(token_idx)
                component.append(component_name)
                component_index.append(component_idx)
                cursor += 1

    nodes_df: dict[str, NDArray[Any]] = {
        "node_index": np.asarray(node_index, dtype=np.int64),
        "layer": np.asarray(layer, dtype=np.int64),
        "token_pos": np.asarray(token_pos, dtype=np.int64),
        "component": np.asarray(component, dtype=object),
        "component_index": np.asarray(component_index, dtype=np.int64),
    }

    examples_df: dict[str, NDArray[Any]] = {
        "example_id": np.asarray(example_ids, dtype=np.int64),
        "slice_id": np.asarray(slice_ids, dtype=object),
        "task": np.asarray(tasks, dtype=object),
        "corruption": np.asarray(corruptions, dtype=object),
        "clean_score": np.asarray(clean_scores, dtype=np.float32),
        "corrupted_score": np.asarray(corrupted_scores, dtype=np.float32),
        "clean_better_than_corrupted": np.asarray(
            clean_better_than_corrupted,
            dtype=bool,
        ),
    }

    return E, nodes_df, examples_df


def plot_patching_heatmap_layer_token(
    E: NDArray[np.float32],
    nodes_df: Any,
    examples_df: Any,
    slice_filter: str | Sequence[str] | None,
    component: str = "resid",
    agg: str = "mean",
    subset: str = _DEFAULT_SUBSET,
    example_filters: Optional[dict[str, Any]] = None,
    add_contours: bool = False,
    contour_percentile: float = 95.0,
    robust_percentile: float = 99.0,
    output_png: Optional[str | Path] = None,
    output_html: Optional[str | Path] = None,
    output_json: Optional[str | Path] = None,
    title: Optional[str] = None,
) -> dict[str, Any]:
    """Plot heatmap of patching effects aggregated by ``(layer, token_pos)``.

    Args:
        E: Effect matrix ``[n_examples, n_nodes]``.
        nodes_df: Tabular structure with columns at least ``layer``,
            ``token_pos``/``token``, and ``component``.
        examples_df: Tabular structure with at least ``slice_id`` and one row
            per example.
        slice_filter: Single slice or list of slices to include.
        component: Component name filter (e.g. ``resid``, ``res``, ``mlp``).
        agg: Aggregation used for output matrix ``M`` (``mean`` or ``median``).
        subset: Example subset selector. ``all`` by default.
            ``clean_correct_corrupted_wrong`` is also supported when
            ``examples_df`` includes ``clean_correct`` and ``corrupted_correct``.
        example_filters: Additional column filters over ``examples_df``.
        add_contours: Draw contour lines over top percentile of ``|M|``.
        contour_percentile: Percentile for contour threshold over ``|M|``.
        robust_percentile: Robust percentile used to set symmetric vmin/vmax.
        output_png: Output path for static PNG.
        output_html: Output path for interactive Plotly HTML.
        output_json: Output path for metadata JSON.
        title: Optional figure title.

    Returns:
        Dict with paths, selected counts, matrices and plotting metadata.
    """
    effect_matrix = np.asarray(E, dtype=np.float32)
    if effect_matrix.ndim != 2:
        raise ValueError(f"E must be 2D. Got shape {effect_matrix.shape}")

    n_examples, n_nodes = effect_matrix.shape
    if n_examples == 0 or n_nodes == 0:
        raise ValueError("E must have at least one example and one node")

    slice_values = _table_column(examples_df, _SLICE_FILTER_COLUMN)
    if len(slice_values) != n_examples:
        raise ValueError(
            "examples_df must have one row per E example "
            f"({len(slice_values)} != {n_examples})"
        )

    selected_slices = _resolve_slice_filter(slice_filter)
    selected_example_mask = np.ones(n_examples, dtype=bool)

    if selected_slices:
        selected_example_mask &= np.isin(slice_values.astype(str), selected_slices)

    selected_example_mask &= _resolve_subset_mask(examples_df, subset)
    selected_example_mask = _apply_example_filters(
        selected_example_mask,
        examples_df,
        example_filters,
    )

    selected_example_indices = np.flatnonzero(selected_example_mask)
    if selected_example_indices.size == 0:
        raise ValueError("No examples left after applying slice/subset filters")

    node_indices = _resolve_node_indices(nodes_df, n_nodes)
    layer_column = _find_first_column(nodes_df, ["layer"])
    token_column = _find_first_column(nodes_df, ["token_pos", "token"])
    component_column = _find_first_column(
        nodes_df,
        ["component", "node_type", "type"],
    )

    layers = _table_column(nodes_df, layer_column).astype(np.int64)
    token_positions = _table_column(nodes_df, token_column).astype(np.int64)
    component_values = _table_column(nodes_df, component_column).astype(str)
    component_values = np.asarray(
        [_normalize_component(value) for value in component_values],
        dtype=object,
    )

    selected_component = _normalize_component(component)
    component_mask = component_values == selected_component
    if not np.any(component_mask):
        raise ValueError(
            f"No nodes found for component '{component}'. "
            f"Available: {sorted(set(component_values.tolist()))}"
        )

    if np.min(layers) < 0 or np.min(token_positions) < 0:
        raise ValueError("layer/token_pos values must be non-negative")

    n_layers = int(np.max(layers)) + 1
    n_token_positions = int(np.max(token_positions)) + 1

    mean_matrix = np.full((n_layers, n_token_positions), np.nan, dtype=np.float32)
    median_matrix = np.full((n_layers, n_token_positions), np.nan, dtype=np.float32)

    selected_effects = effect_matrix[selected_example_indices]

    for layer_idx in range(n_layers):
        for token_idx in range(n_token_positions):
            node_mask = component_mask & (layers == layer_idx) & (token_positions == token_idx)
            node_columns = node_indices[node_mask]
            if node_columns.size == 0:
                continue

            values = selected_effects[:, node_columns].reshape(-1)
            mean_matrix[layer_idx, token_idx] = float(np.mean(values))
            median_matrix[layer_idx, token_idx] = float(np.median(values))

    aggregation = agg.strip().lower()
    if aggregation not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")

    matrix = mean_matrix if aggregation == "mean" else median_matrix
    finite_values = np.abs(matrix[np.isfinite(matrix)])

    if finite_values.size == 0:
        robust_limit = 1.0
    else:
        robust_limit = float(np.percentile(finite_values, robust_percentile))
        if not np.isfinite(robust_limit) or robust_limit <= 0:
            robust_limit = float(np.max(finite_values))
        if robust_limit <= 0:
            robust_limit = 1e-6

    vmin = -robust_limit
    vmax = robust_limit
    center = 0.0

    display_matrix = np.nan_to_num(matrix, nan=0.0)

    contour_threshold: float | None = None
    if add_contours:
        contour_threshold = float(
            np.percentile(np.abs(display_matrix), contour_percentile)
        )
        if not np.isfinite(contour_threshold) or contour_threshold <= 0:
            contour_threshold = None

    if selected_slices:
        slice_tag = "__".join(_sanitize_slug(value) for value in selected_slices)
    else:
        slice_tag = "all"

    output_png_path = Path(output_png) if output_png else Path(
        f"reports/figures/A_heatmap_layer_token__{slice_tag}.png"
    )
    output_html_path = Path(output_html) if output_html else Path(
        f"reports/html/A_heatmap_layer_token__{slice_tag}.html"
    )
    output_json_path = Path(output_json) if output_json else Path(
        f"reports/metadata/A_heatmap_layer_token__{slice_tag}.json"
    )

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(
        display_matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )
    if add_contours and contour_threshold is not None:
        axis.contour(
            np.abs(display_matrix),
            levels=[contour_threshold],
            colors="black",
            linewidths=0.8,
        )

    axis.set_title(
        title
        or (
            "Patching effects aggregated by (layer, token_pos) "
            f"[{aggregation}]"
        )
    )
    axis.set_xlabel("token_pos")
    axis.set_ylabel("layer")
    figure.colorbar(image, ax=axis, label=f"{aggregation} patch effect")
    figure.tight_layout()
    figure.savefig(output_png_path, dpi=150)
    plt.close(figure)

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Plotly is required for interactive HTML output. "
            "Install with: pip install plotly"
        ) from exc

    plotly_figure = go.Figure()
    plotly_figure.add_trace(
        go.Heatmap(
            z=display_matrix,
            colorscale="RdBu",
            zmid=center,
            zmin=vmin,
            zmax=vmax,
            colorbar={"title": f"{aggregation} effect"},
        )
    )
    if add_contours and contour_threshold is not None:
        plotly_figure.add_trace(
            go.Contour(
                z=np.abs(display_matrix),
                contours={
                    "start": contour_threshold,
                    "end": contour_threshold,
                    "size": 1,
                    "coloring": "none",
                },
                line={"color": "black", "width": 1},
                showscale=False,
                hoverinfo="skip",
            )
        )

    plotly_figure.update_layout(
        title=axis.get_title(),
        xaxis_title="token_pos",
        yaxis_title="layer",
        template="plotly_white",
    )
    plotly_figure.update_yaxes(autorange="reversed")
    plotly_figure.write_html(str(output_html_path), include_plotlyjs="cdn")

    metadata = {
        "slice_filter": selected_slices,
        "component": selected_component,
        "aggregation": aggregation,
        "subset": subset,
        "example_filters": example_filters or {},
        "num_examples_total": int(n_examples),
        "num_examples_selected": int(selected_example_indices.size),
        "num_nodes_total": int(n_nodes),
        "num_nodes_component": int(np.sum(component_mask)),
        "matrix_shape": [int(n_layers), int(n_token_positions)],
        "vmin": float(vmin),
        "vmax": float(vmax),
        "center": center,
        "robust_percentile": float(robust_percentile),
        "contours": {
            "enabled": bool(add_contours),
            "percentile": float(contour_percentile),
            "threshold": contour_threshold,
        },
        "outputs": {
            "png": str(output_png_path),
            "html": str(output_html_path),
            "json": str(output_json_path),
        },
    }

    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return {
        "matrix": matrix,
        "mean_matrix": mean_matrix,
        "median_matrix": median_matrix,
        "metadata": metadata,
        "output_png": output_png_path,
        "output_html": output_html_path,
        "output_json": output_json_path,
    }


def save_slice_heatmaps(
    dataset: PatchEffectDataset,
    output_dir: Path,
    token_labels_by_slice: Optional[dict[SliceLabel, list[str]]] = None,
) -> list[Path]:
    """Save one average residual-effect heatmap per slice.

    Returns a list of created image paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    residual_index = _get_residual_component_index(dataset)
    created_paths: list[Path] = []

    for slice_label in sorted(
        dataset.get_slices(), key=lambda label: (label.task, label.corruption)
    ):
        tensors = dataset.get_by_slice(slice_label)
        if not tensors:
            continue

        _, min_tokens, _ = dataset.get_common_dimensions(slice_label)
        stacked = np.stack(
            [tensor.effects[:, :min_tokens, residual_index] for tensor in tensors],
            axis=0,
        )
        mean_effects = np.mean(stacked, axis=0)

        figure, axis = plt.subplots(figsize=(8, 4))
        image = axis.imshow(mean_effects, aspect="auto", cmap="coolwarm")
        axis.set_title(f"Mean patch effects ({slice_label.corruption})")
        axis.set_xlabel("Token")
        axis.set_ylabel("Layer")

        labels_for_slice = None
        if token_labels_by_slice is not None:
            labels_for_slice = token_labels_by_slice.get(slice_label)
        if labels_for_slice:
            labels = _build_tick_labels(labels_for_slice[:min_tokens])
            axis.set_xticks(np.arange(len(labels)))
            axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)

        figure.colorbar(image, ax=axis, label="Effect")
        figure.tight_layout()

        file_name = f"heatmap_{slice_label.task}_{slice_label.corruption}.png"
        output_path = output_dir / file_name
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        created_paths.append(output_path)

    return created_paths


def save_pca_embeddings(
    feature_matrix: WLFeatureMatrix,
    output_path: Path,
) -> Path:
    """Save a 2D PCA projection of graph embeddings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = feature_matrix.to_matrix()
    if matrix.shape[0] < 2:
        raise ValueError("Need at least 2 graphs to create PCA plot")

    n_components = min(2, matrix.shape[0], matrix.shape[1])
    reduced = PCA(
        n_components=n_components,
        random_state=42,
    ).fit_transform(matrix)

    if n_components == 1:
        reduced = np.concatenate(
            [reduced, np.zeros((reduced.shape[0], 1), dtype=reduced.dtype)],
            axis=1,
        )

    labels = [label.corruption for label in feature_matrix.slice_labels]
    unique_labels = sorted(set(labels))

    figure, axis = plt.subplots(figsize=(7, 5))
    for label in unique_labels:
        indices = [index for index, value in enumerate(labels) if value == label]
        points = reduced[indices]
        axis.scatter(points[:, 0], points[:, 1], label=label, alpha=0.8)

    axis.set_title("PCA of WL Graph Embeddings")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    return output_path
