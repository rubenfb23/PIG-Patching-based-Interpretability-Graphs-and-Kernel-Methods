"""Visualization utilities for pipeline artifacts.

This module generates the three output figures expected by the project:
- ``heatmap_ioi_abba.png``
- ``heatmap_ioi_name_swap.png``
- ``pca_embeddings.png``
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from pig.embeddings import WLFeatureMatrix
from pig.patching import NODE_TYPE_RES, PatchEffectDataset
from pig.prompts import SliceLabel


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
            [
                tensor.effects[:, :min_tokens, residual_index]
                for tensor in tensors
            ],
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
        indices = [
            index for index, value in enumerate(labels) if value == label
        ]
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
