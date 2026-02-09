#!/usr/bin/env python3
"""Validation script for Story 5.2 - Quantum Fidelity Kernel.

This script verifies:
1. Kernel matrix computation
2. Precomputed-kernel SVM training
3. Sensible variation with shots and depth
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from pig.embeddings import WLEmbedding, WLFeatureMatrix
from pig.prompts import SliceLabel
from pig.quantum import compute_quantum_kernel_matrix, train_quantum_kernel_baseline


def _make_feature_matrix(
    num_graphs: int = 12, num_features: int = 10
) -> WLFeatureMatrix:
    rng = np.random.default_rng(123)
    vocabulary = [f"f{i}" for i in range(num_features)]
    embeddings = []
    slice_labels = []
    for i in range(num_graphs):
        shift = 3 if i % 2 == 0 else 0
        features = {
            f"f{j}": int(rng.integers(0, 5) + shift) for j in range(num_features)
        }
        label = SliceLabel(
            task="ioi",
            corruption="name_swap" if i % 2 == 0 else "abba",
        )
        embeddings.append(
            WLEmbedding(features=features, graph_id=str(label), depth=3)
        )
        slice_labels.append(label)
    return WLFeatureMatrix(
        embeddings=embeddings,
        vocabulary=vocabulary,
        slice_labels=slice_labels,
    )


def main() -> None:
    print("=" * 60)
    print("STORY 5.2 VALIDATION - Quantum Fidelity Kernel")
    print("=" * 60)

    feature_matrix = _make_feature_matrix()

    # Basic kernel computation
    K, _ = compute_quantum_kernel_matrix(
        feature_matrix,
        n_qubits=4,
        depth=2,
        shots=500,
        reduction="pca",
        random_state=42,
    )
    print(f"  Kernel matrix shape: {K.shape}")
    if K.shape[0] == K.shape[1] == feature_matrix.num_graphs:
        print("  [PASS] Kernel matrix computed")
    else:
        print("  [FAIL] Kernel matrix shape mismatch")

    # Sweep shots and depth
    shots_list = [200, 500, 2000, 5000]
    depth_list = [1, 2, 3]

    for depth in depth_list:
        print("\n" + "-" * 40)
        print(f"  Depth D = {depth}")
        print("-" * 40)
        for shots in shots_list:
            _, cv_results, _ = train_quantum_kernel_baseline(
                feature_matrix,
                n_qubits=4,
                depth=depth,
                shots=shots,
                reduction="pca",
                random_state=42,
            )
            print(
                f"  Shots={shots:4d} | "
                f"CV acc={cv_results['accuracy_mean']:.2%} "
                f"± {cv_results['accuracy_std']:.2%}"
            )

    print(
        "\n  [INFO] If accuracies vary across shots/depth, "
        "the kernel is sensitive as expected."
    )


if __name__ == "__main__":
    main()
