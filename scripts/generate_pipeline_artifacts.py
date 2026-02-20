#!/usr/bin/env python3
"""Generate output image artifacts for the current model pipeline run."""

from __future__ import annotations
from pathlib import Path

from pig.embeddings import compute_wl_features_from_list
from pig.graph import create_graph_builder
from pig.model import create_model_from_env
from pig.patching import PatchEffectComputer, PatchEffectDataset
from pig.prompts import ABBACorruption, IOIGenerator
from pig.visualization import (
    plot_patching_heatmap_layer_token,
    prepare_patching_heatmap_inputs,
    save_pca_embeddings,
    save_slice_heatmaps,
)

def _build_dataset() -> tuple[PatchEffectDataset, object]:
    model = create_model_from_env()

    generator_swap = IOIGenerator(seed=42)
    generator_abba = IOIGenerator(corruption=ABBACorruption(), seed=42)
    pairs = generator_swap.generate_batch(8) + generator_abba.generate_batch(8)

    computer = PatchEffectComputer(model)
    dataset = PatchEffectDataset()
    for index, pair in enumerate(pairs):
        print(
            f"\r  Computing artifact effects: {index + 1}/{len(pairs)}",
            end="",
        )
        dataset.add(computer.compute_single(pair))
    print()

    return dataset, model


def _build_token_labels_by_slice(
    dataset: PatchEffectDataset,
    model,
) -> dict[SliceLabel, list[str]]:
    labels: dict[SliceLabel, list[str]] = {}
    for slice_label in dataset.get_slices():
        tensors = dataset.get_by_slice(slice_label)
        if not tensors:
            continue
        example_prompt = tensors[0].prompt_pair.x_crp
        labels[slice_label] = model.get_prompt_tokens(example_prompt)
    return labels


def main() -> int:
    print("=" * 60)
    print("GENERATE PIPELINE ARTIFACTS")
    print("=" * 60)

    dataset, model = _build_dataset()
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    token_labels_by_slice = _build_token_labels_by_slice(dataset, model)

    heatmaps = save_slice_heatmaps(
        dataset=dataset,
        output_dir=output_dir,
        token_labels_by_slice=token_labels_by_slice,
    )
    for path in heatmaps:
        print(f"  Saved: {path.name}")

    builder = create_graph_builder(k=5, enforce_direction=True)
    graphs_with_labels = builder.build_per_example(dataset)
    feature_matrix = compute_wl_features_from_list(graphs_with_labels, depth=3)

    pca_path = save_pca_embeddings(
        feature_matrix,
        output_dir / "pca_embeddings.png",
    )
    print(f"  Saved: {pca_path.name}")

    E, nodes_df, examples_df = prepare_patching_heatmap_inputs(dataset)
    for slice_label in sorted(
        dataset.get_slices(),
        key=lambda label: (label.task, label.corruption),
    ):
        heatmap_result = plot_patching_heatmap_layer_token(
            E,
            nodes_df,
            examples_df,
            slice_filter=str(slice_label),
            component="resid",
            agg="mean",
        )
        print(
            "  Saved: "
            f"{heatmap_result['output_png'].name}, "
            f"{heatmap_result['output_html'].name}, "
            f"{heatmap_result['output_json'].name}"
        )

    if len(heatmaps) < 2:
        print("  [FAIL] Expected at least two slice heatmaps")
        return 1

    print("  [PASS] Output artifacts generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
