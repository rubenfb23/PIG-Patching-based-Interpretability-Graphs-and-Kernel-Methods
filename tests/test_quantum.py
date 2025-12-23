"""Unit tests for pig.quantum module."""

import numpy as np

from pig.embeddings import WLEmbedding, WLFeatureMatrix
from pig.prompts import SliceLabel
from pig.quantum import (
    QuantumFeatureMap,
    QuantumFeatureReducer,
    QuantumKernelClassifier,
    compute_fidelity_kernel_matrix,
    compute_quantum_kernel_matrix,
)


def _make_feature_matrix(
    num_graphs: int = 6, num_features: int = 8
) -> WLFeatureMatrix:
    rng = np.random.default_rng(42)
    vocabulary = [f"f{i}" for i in range(num_features)]
    embeddings = []
    slice_labels = []
    for i in range(num_graphs):
        features = {f"f{j}": int(rng.integers(0, 5)) for j in range(num_features)}
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


def test_feature_reducer_shapes():
    X = np.random.randn(10, 6).astype(np.float32)
    reducer = QuantumFeatureReducer(method="pca", n_components=3, random_state=0)
    Z = reducer.fit_transform(X)
    assert Z.shape == (10, 3)


def test_feature_map_state_norm():
    feature_map = QuantumFeatureMap(n_qubits=3, depth=1)
    z = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    state = feature_map.state(z)
    norm = np.sum(np.abs(state) ** 2)
    assert np.isclose(norm, 1.0, atol=1e-6)


def test_kernel_matrix_properties():
    feature_map = QuantumFeatureMap(n_qubits=2, depth=1)
    Z = np.array(
        [
            [0.0, 0.0],
            [np.pi, np.pi],
            [0.0, np.pi],
        ],
        dtype=np.float32,
    )
    K = compute_fidelity_kernel_matrix(Z, feature_map)
    assert K.shape == (3, 3)
    assert np.allclose(K, K.T, atol=1e-6)
    assert np.allclose(np.diag(K), 1.0, atol=1e-6)
    assert np.all(K >= 0.0) and np.all(K <= 1.0)


def test_quantum_classifier_separable():
    n_qubits = 3
    feature_map = QuantumFeatureMap(n_qubits=n_qubits, depth=1)

    Z0 = np.zeros((10, n_qubits), dtype=np.float32)
    Z1 = np.pi * np.ones((10, n_qubits), dtype=np.float32)
    Z = np.vstack([Z0, Z1])
    y = np.array([0] * 10 + [1] * 10)

    K = compute_fidelity_kernel_matrix(Z, feature_map)
    clf = QuantumKernelClassifier(random_state=42)
    cv = clf.cross_validate(K, y, cv=3)
    assert cv["accuracy_mean"] > 0.8

    clf.fit(K, y)
    preds = clf.predict(K)
    assert np.mean(preds == y) > 0.8


def test_compute_quantum_kernel_matrix():
    feature_matrix = _make_feature_matrix()
    K, reducer = compute_quantum_kernel_matrix(
        feature_matrix, n_qubits=3, depth=1, shots=None, reduction="pca"
    )
    assert K.shape == (feature_matrix.num_graphs, feature_matrix.num_graphs)
    assert np.allclose(np.diag(K), 1.0, atol=1e-6)
    assert reducer.n_components == 3
