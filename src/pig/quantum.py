"""Quantum feature maps and fidelity kernels.

Implements a lightweight, in-repo quantum simulator for small feature maps
to avoid heavyweight dependencies while enabling quantum kernel experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import SVC

from pig.embeddings import WLFeatureMatrix
from pig.prompts import SliceLabel


@dataclass
class QuantumFeatureReducer:
    """Dimensionality reduction for quantum feature maps."""

    method: Literal["pca", "random"] = "pca"
    n_components: int = 4
    random_state: int = 42
    normalize: bool = True

    _scaler: Optional[StandardScaler] = None
    _reducer: Optional[object] = None

    def fit_transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Fit the reducer and transform features."""
        if self.n_components > X.shape[1]:
            raise ValueError("n_components must be <= number of features")

        if self.normalize:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        if self.method == "pca":
            self._reducer = PCA(
                n_components=self.n_components, random_state=self.random_state
            )
        elif self.method == "random":
            self._reducer = GaussianRandomProjection(
                n_components=self.n_components, random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown reduction method: {self.method}")

        Z = self._reducer.fit_transform(X)
        return Z.astype(np.float32)

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Transform features using the fitted reducer."""
        if self._reducer is None:
            raise RuntimeError("Reducer not fitted. Call fit_transform first.")

        if self.normalize and self._scaler is not None:
            X = self._scaler.transform(X)

        Z = self._reducer.transform(X)
        return Z.astype(np.float32)


@dataclass
class QuantumFeatureMap:
    """Simple quantum feature map with entangling CZ layers."""

    n_qubits: int = 4
    depth: int = 2
    feature_scale: float = 1.0

    def _apply_single_qubit_gate(
        self,
        state: NDArray[np.complex128],
        gate: NDArray[np.complex128],
        qubit: int,
    ) -> NDArray[np.complex128]:
        """Apply a single-qubit gate to a statevector."""
        n = self.n_qubits
        state_tensor = state.reshape([2] * n)
        state_tensor = np.moveaxis(state_tensor, qubit, 0)
        updated = np.tensordot(gate, state_tensor, axes=[1, 0])
        updated = np.moveaxis(updated, 0, qubit)
        return updated.reshape(-1)

    def _apply_two_qubit_gate(
        self,
        state: NDArray[np.complex128],
        gate: NDArray[np.complex128],
        qubit_a: int,
        qubit_b: int,
    ) -> NDArray[np.complex128]:
        """Apply a two-qubit gate to a statevector."""
        if qubit_a == qubit_b:
            raise ValueError("Two-qubit gate requires distinct qubits")

        n = self.n_qubits
        state_tensor = state.reshape([2] * n)
        state_tensor = np.moveaxis(state_tensor, [qubit_a, qubit_b], [0, 1])
        flat = state_tensor.reshape(4, -1)
        updated = gate @ flat
        updated = updated.reshape([2, 2] + [2] * (n - 2))
        updated = np.moveaxis(updated, [0, 1], [qubit_a, qubit_b])
        return updated.reshape(-1)

    def _gate_h(self) -> NDArray[np.complex128]:
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        return inv_sqrt2 * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)

    def _gate_rz(self, theta: float) -> NDArray[np.complex128]:
        half = theta / 2.0
        return np.array(
            [[np.exp(-1j * half), 0.0], [0.0, np.exp(1j * half)]],
            dtype=np.complex128,
        )

    def _gate_rx(self, theta: float) -> NDArray[np.complex128]:
        half = theta / 2.0
        return np.array(
            [
                [np.cos(half), -1j * np.sin(half)],
                [-1j * np.sin(half), np.cos(half)],
            ],
            dtype=np.complex128,
        )

    def _gate_cz(self) -> NDArray[np.complex128]:
        return np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)

    def state(self, features: NDArray[np.float32]) -> NDArray[np.complex128]:
        """Compute the statevector for a feature vector."""
        if features.shape[0] != self.n_qubits:
            raise ValueError(
                "Feature vector length must match number of qubits"
            )

        state = np.zeros(2**self.n_qubits, dtype=np.complex128)
        state[0] = 1.0

        h_gate = self._gate_h()
        cz_gate = self._gate_cz()

        for qubit in range(self.n_qubits):
            state = self._apply_single_qubit_gate(state, h_gate, qubit)

        scaled = features * self.feature_scale

        for _ in range(self.depth):
            for qubit in range(self.n_qubits):
                state = self._apply_single_qubit_gate(
                    state, self._gate_rz(float(scaled[qubit])), qubit
                )
                state = self._apply_single_qubit_gate(
                    state, self._gate_rx(float(scaled[qubit])), qubit
                )
            for qubit in range(self.n_qubits - 1):
                state = self._apply_two_qubit_gate(
                    state, cz_gate, qubit, qubit + 1
                )

        return state


def compute_fidelity_kernel_matrix(
    Z: NDArray[np.float32],
    feature_map: QuantumFeatureMap,
    shots: Optional[int] = None,
    seed: int = 42,
) -> NDArray[np.float32]:
    """Compute a quantum fidelity kernel matrix from reduced features."""
    states = []
    for row in Z:
        state = feature_map.state(row.astype(np.float32))
        states.append(state)

    state_matrix = np.vstack(states)
    overlaps = state_matrix.conj() @ state_matrix.T
    fidelity = np.abs(overlaps) ** 2
    fidelity = np.clip(fidelity, 0.0, 1.0)

    if shots is None or shots <= 0:
        return fidelity.astype(np.float32)

    rng = np.random.default_rng(seed)
    n = fidelity.shape[0]
    noisy = np.zeros_like(fidelity)

    for i in range(n):
        noisy[i, i] = 1.0
        for j in range(i + 1, n):
            p0 = 0.5 * (1.0 + fidelity[i, j])
            count = rng.binomial(shots, p0)
            estimate = 2.0 * count / shots - 1.0
            estimate = float(np.clip(estimate, 0.0, 1.0))
            noisy[i, j] = estimate
            noisy[j, i] = estimate

    return noisy.astype(np.float32)


@dataclass
class QuantumClassificationResult:
    """Results from a quantum-kernel classification experiment."""

    accuracy: float
    auc: Optional[float]
    predictions: NDArray[np.int32]
    true_labels: NDArray[np.int32]
    kernel_type: str

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "kernel_type": self.kernel_type,
        }


class QuantumKernelClassifier:
    """SVM classifier for precomputed quantum kernels."""

    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self._svm: Optional[SVC] = None
        self._label_encoder: Optional[LabelEncoder] = None

    def _create_svm(self) -> SVC:
        return SVC(
            kernel="precomputed",
            C=self.C,
            random_state=self.random_state,
            probability=True,
        )

    def fit(
        self,
        K_train: NDArray[np.float32],
        y: list[SliceLabel] | NDArray,
    ) -> "QuantumKernelClassifier":
        if isinstance(y, list) and isinstance(y[0], SliceLabel):
            self._label_encoder = LabelEncoder()
            y_encoded = self._label_encoder.fit_transform(
                [str(label) for label in y]
            )
        else:
            y_encoded = np.array(y)

        self._svm = self._create_svm()
        self._svm.fit(K_train, y_encoded)
        return self

    def predict(self, K_test: NDArray[np.float32]) -> NDArray[np.int32]:
        if self._svm is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        return self._svm.predict(K_test)

    def predict_proba(self, K_test: NDArray[np.float32]) -> NDArray[np.float64]:
        if self._svm is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        return self._svm.predict_proba(K_test)

    def evaluate(
        self,
        K_test: NDArray[np.float32],
        y: list[SliceLabel] | NDArray,
    ) -> QuantumClassificationResult:
        if isinstance(y, list) and isinstance(y[0], SliceLabel):
            if self._label_encoder is None:
                raise RuntimeError("Classifier not fitted")
            y_encoded = self._label_encoder.transform(
                [str(label) for label in y]
            )
        else:
            y_encoded = np.array(y)

        predictions = self.predict(K_test)
        accuracy = accuracy_score(y_encoded, predictions)

        auc = None
        n_classes = len(np.unique(y_encoded))
        if n_classes == 2:
            proba = self.predict_proba(K_test)[:, 1]
            auc = roc_auc_score(y_encoded, proba)

        return QuantumClassificationResult(
            accuracy=float(accuracy),
            auc=float(auc) if auc is not None else None,
            predictions=predictions,
            true_labels=y_encoded,
            kernel_type="quantum",
        )

    def cross_validate(
        self,
        K: NDArray[np.float32],
        y: list[SliceLabel] | NDArray,
        cv: int = 5,
    ) -> dict:
        if isinstance(y, list) and isinstance(y[0], SliceLabel):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(
                [str(label) for label in y]
            )
        else:
            y_encoded = np.array(y)

        splitter = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=self.random_state
        )
        scores = []

        for train_idx, test_idx in splitter.split(K, y_encoded):
            K_train = K[np.ix_(train_idx, train_idx)]
            K_test = K[np.ix_(test_idx, train_idx)]

            svm = self._create_svm()
            svm.fit(K_train, y_encoded[train_idx])
            preds = svm.predict(K_test)
            acc = accuracy_score(y_encoded[test_idx], preds)
            scores.append(acc)

        return {
            "accuracy_mean": float(np.mean(scores)),
            "accuracy_std": float(np.std(scores)),
            "kernel_type": "quantum",
            "cv_folds": cv,
        }


def compute_quantum_kernel_matrix(
    feature_matrix: WLFeatureMatrix,
    n_qubits: int = 4,
    depth: int = 2,
    shots: Optional[int] = None,
    reduction: Literal["pca", "random"] = "pca",
    random_state: int = 42,
    feature_scale: float = 1.0,
) -> tuple[NDArray[np.float32], QuantumFeatureReducer]:
    """Compute a quantum kernel matrix from WL features."""
    X = feature_matrix.to_matrix()
    reducer = QuantumFeatureReducer(
        method=reduction,
        n_components=n_qubits,
        random_state=random_state,
    )
    Z = reducer.fit_transform(X)

    feature_map = QuantumFeatureMap(
        n_qubits=n_qubits, depth=depth, feature_scale=feature_scale
    )
    K = compute_fidelity_kernel_matrix(
        Z, feature_map=feature_map, shots=shots, seed=random_state
    )
    return K, reducer


def train_quantum_kernel_baseline(
    feature_matrix: WLFeatureMatrix,
    n_qubits: int = 4,
    depth: int = 2,
    shots: Optional[int] = None,
    reduction: Literal["pca", "random"] = "pca",
    random_state: int = 42,
    feature_scale: float = 1.0,
) -> tuple[QuantumKernelClassifier, dict, NDArray[np.float32]]:
    """Train an SVM classifier with a quantum fidelity kernel."""
    K, _ = compute_quantum_kernel_matrix(
        feature_matrix=feature_matrix,
        n_qubits=n_qubits,
        depth=depth,
        shots=shots,
        reduction=reduction,
        random_state=random_state,
        feature_scale=feature_scale,
    )

    y = feature_matrix.slice_labels
    classifier = QuantumKernelClassifier(random_state=random_state)
    cv_results = classifier.cross_validate(K, y)
    classifier.fit(K, y)

    return classifier, cv_results, K
