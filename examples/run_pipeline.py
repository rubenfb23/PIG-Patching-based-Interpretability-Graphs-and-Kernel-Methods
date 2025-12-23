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
    handlers=[logging.StreamHandler(sys.stdout)]
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
    logger.error("Make sure you have installed the package in editable mode: pip install -e .")
    sys.exit(1)

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
