#!/usr/bin/env python3
"""
End-to-end example of the PIG pipeline.

This script demonstrates:
1. Loading a model
2. Generating IOI prompt pairs
3. Computing patch effects
4. Building graphs
5. Computing WL embeddings
6. Training a classical kernel baseline
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

try:
    from pig.model import HookedModel
    from pig.prompts import create_ioi_dataset
    from pig.patching import compute_patch_effects
    from pig.graph import GraphBuilder
    from pig.embeddings import compute_wl_features, compute_wl_features_from_list
    from pig.kernels import train_classical_baseline
except ImportError as e:
    logger.error(f"Could not import 'pig': {e}")
    logger.error(
        "Make sure you have installed the package in editable mode: pip install -e ."
    )
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.decomposition import PCA

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not found. Visualizations will be skipped.")


def plot_average_heatmap(effects, filename="heatmap.png"):
    """Plot average patch effect heatmap for each slice."""
    if not VISUALIZATION_AVAILABLE:
        return

    logger.info("   Generating heatmaps...")

    # Aggregate effects by slice
    slice_effects = {}
    for tensor in effects:
        label = str(tensor.prompt_pair.slice_label)
        if label not in slice_effects:
            slice_effects[label] = []
        slice_effects[label].append(tensor.effects)

    # Plot average heatmap for each slice
    for label, tensors in slice_effects.items():
        # Handle variable shapes by padding to max length
        max_tokens = max(t.shape[1] for t in tensors)
        num_layers = tensors[0].shape[0]

        padded_tensors = []
        for t in tensors:
            pad_width = max_tokens - t.shape[1]
            if pad_width > 0:
                # Pad on the right with NaNs
                padded = np.pad(
                    t, ((0, 0), (0, pad_width)), mode="constant", constant_values=np.nan
                )
                padded_tensors.append(padded)
            else:
                padded_tensors.append(t)

        # Compute mean ignoring NaNs
        avg_effects = np.nanmean(np.stack(padded_tensors), axis=0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_effects, cmap="RdBu_r", center=0)
        plt.title(f"Average Patch Effect: {label}")
        plt.xlabel("Token Position")
        plt.ylabel("Layer")

        # Sanitize filename
        safe_label = label.replace(":", "_").replace(" ", "_").replace("/", "_")
        out_file = f"{filename.split('.')[0]}_{safe_label}.png"

        plt.savefig(out_file)
        plt.close()
        logger.info(f"   Saved heatmap to {out_file}")


def plot_pca_embeddings(features, filename="pca.png"):
    """Plot PCA of graph embeddings."""
    if not VISUALIZATION_AVAILABLE:
        return

    logger.info("   Generating PCA plot...")

    X = features.to_matrix()
    # Extract labels from slice_labels
    labels = [str(l.corruption) for l in features.slice_labels]

    if X.shape[0] < 2:
        logger.warning("Not enough samples for PCA.")
        return

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, style=labels, s=100)
    plt.title("PCA of Graph Embeddings (WL Features)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()
    logger.info(f"   Saved PCA plot to {filename}")


def main():
    logger.info("Starting PIG pipeline example...")

    # 1. Load a model with activation hooks
    logger.info("1. Loading model (gpt2)...")
    # Using cpu for example to ensure it runs everywhere, but auto-detection is default
    model = HookedModel(model_name="gpt2")

    # 2. Generate clean/corrupted prompt pairs
    logger.info("2. Generating IOI dataset...")
    # Generate two classes of data to allow classification
    dataset_swap = create_ioi_dataset(n_examples=10, corruption="name_swap", seed=42)
    dataset_abba = create_ioi_dataset(n_examples=10, corruption="abba", seed=43)
    dataset = dataset_swap + dataset_abba
    logger.info(f"   Generated {len(dataset)} examples (10 name_swap, 10 abba).")

    # 3. Compute patch effects for all (layer, token) positions
    logger.info("3. Computing patch effects...")
    effects = compute_patch_effects(model, dataset)
    logger.info(f"   Computed effects for {len(effects)} slices.")

    # Visualization: Heatmaps
    plot_average_heatmap(effects, filename="examples/heatmap.png")

    # 4. Build graphs from patch effects
    logger.info("4. Building graphs (per example)...")
    builder = GraphBuilder(k=5, enforce_direction=True)
    # Use build_per_example to treat each example as a graph
    graphs_with_labels = builder.build_per_example(effects)
    logger.info(f"   Built {len(graphs_with_labels)} graphs.")

    # 5. Compute WL graph embeddings
    logger.info("5. Computing WL embeddings...")
    features = compute_wl_features_from_list(graphs_with_labels, depth=3)
    logger.info(f"   Feature matrix shape: {features.to_matrix().shape}")

    # Visualization: PCA
    plot_pca_embeddings(features, filename="examples/pca_embeddings.png")

    # 6. Train a classical kernel baseline
    logger.info("6. Training classical baseline...")
    # Note: With only 20 examples, accuracy might be unstable/low, but this proves the pipeline runs.
    classifier, results = train_classical_baseline(features, kernel="rbf")

    logger.info("-" * 40)
    logger.info(f"Cross-validation accuracy: {results['accuracy_mean']:.2%}")
    logger.info("-" * 40)
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
