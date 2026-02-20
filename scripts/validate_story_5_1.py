#!/usr/bin/env python3
"""Validation script for Story 5.1 - Quantum Feature Map Circuit.

This script verifies:
1. Dimensionality reduction to n_qubits
2. Quantum feature map state preparation works
3. State vectors are normalized
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from pig.embeddings import WLEmbedding, WLFeatureMatrix
from pig.prompts import SliceLabel
from pig.quantum import QuantumFeatureMap, QuantumFeatureReducer


def _make_feature_matrix(
    num_graphs: int = 6, num_features: int = 8
) -> WLFeatureMatrix:
    rng = np.random.default_rng(42)
    vocabulary = [f"f{i}" for i in range(num_features)]
    embeddings = []
    slice_labels = []
    for i in range(num_graphs):
        features = {
            f"f{j}": int(rng.integers(0, 5)) for j in range(num_features)
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
    print("STORY 5.1 VALIDATION - Quantum Feature Map Circuit")
    print("=" * 60)

    feature_matrix = _make_feature_matrix()
    X = feature_matrix.to_matrix()

    reducer = QuantumFeatureReducer(method="pca", n_components=4, random_state=42)
    Z = reducer.fit_transform(X)

    print(f"  Reduced shape: {Z.shape}")
    if Z.shape[1] == 4:
        print("  [PASS] Dimensionality reduction to n_qubits works")
    else:
        print("  [FAIL] Dimensionality reduction failed")

    feature_map = QuantumFeatureMap(n_qubits=4, depth=2)
    state = feature_map.state(Z[0])
    norm = np.sum(np.abs(state) ** 2)

    print(f"  Statevector norm: {norm:.6f}")
    if np.isclose(norm, 1.0, atol=1e-6):
        print("  [PASS] Feature map produces normalized states")
    else:
        print("  [FAIL] Statevector not normalized")


if __name__ == "__main__":
    main()
