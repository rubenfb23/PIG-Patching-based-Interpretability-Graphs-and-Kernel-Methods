"""Classical kernel methods for graph classification.

This module provides SVM-based classifiers using classical kernels
(linear, RBF) on WL graph embeddings as a baseline for comparison
with quantum kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from pig.embeddings import WLFeatureMatrix
from pig.prompts import SliceLabel


@dataclass
class ClassificationResult:
    """Results from a classification experiment.

    Attributes:
        accuracy: Classification accuracy
        auc: Area under ROC curve (for binary classification)
        predictions: Predicted labels
        true_labels: True labels
        kernel_type: Type of kernel used
    """

    accuracy: float
    auc: Optional[float]
    predictions: NDArray[np.int32]
    true_labels: NDArray[np.int32]
    kernel_type: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "kernel_type": self.kernel_type,
        }


@dataclass
class LearningCurveResult:
    """Results from a learning curve experiment.

    Attributes:
        train_sizes: Number of training examples used
        train_scores: Training accuracy at each size
        test_scores: Test accuracy at each size
        kernel_type: Type of kernel used
    """

    train_sizes: NDArray[np.int32]
    train_scores_mean: NDArray[np.float64]
    train_scores_std: NDArray[np.float64]
    test_scores_mean: NDArray[np.float64]
    test_scores_std: NDArray[np.float64]
    kernel_type: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "train_sizes": self.train_sizes.tolist(),
            "train_scores_mean": self.train_scores_mean.tolist(),
            "test_scores_mean": self.test_scores_mean.tolist(),
            "kernel_type": self.kernel_type,
        }


class ClassicalKernelClassifier:
    """SVM classifier with classical kernels.

    Supports linear and RBF kernels on WL graph embeddings.
    """

    def __init__(
        self,
        kernel: Literal["linear", "rbf"] = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        random_state: int = 42,
        normalize: bool = True,
    ):
        """Initialize the classifier.

        Args:
            kernel: Kernel type ("linear" or "rbf")
            C: SVM regularization parameter
            gamma: RBF kernel coefficient (for rbf kernel)
            random_state: Random seed for reproducibility
            normalize: Whether to standardize features
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.normalize = normalize

        self._scaler: Optional[StandardScaler] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._svm: Optional[SVC] = None

    def _create_svm(self) -> SVC:
        """Create a new SVM instance."""
        return SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            random_state=self.random_state,
            probability=True,  # Needed for AUC
        )

    def fit(
        self,
        X: NDArray[np.float32],
        y: list[SliceLabel] | NDArray,
    ) -> "ClassicalKernelClassifier":
        """Fit the classifier.

        Args:
            X: Feature matrix of shape [n_samples, n_features]
            y: Labels (SliceLabels or encoded integers)

        Returns:
            self
        """
        # Encode labels if needed
        if isinstance(y, list) and isinstance(y[0], SliceLabel):
            self._label_encoder = LabelEncoder()
            y_encoded = self._label_encoder.fit_transform([str(label) for label in y])
        else:
            y_encoded = np.array(y)

        # Normalize features
        if self.normalize:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = X

        # Fit SVM
        self._svm = self._create_svm()
        self._svm.fit(X_scaled, y_encoded)

        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        """Predict labels for new samples.

        Args:
            X: Feature matrix of shape [n_samples, n_features]

        Returns:
            Predicted labels
        """
        if self._svm is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        if self.normalize and self._scaler is not None:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        return self._svm.predict(X_scaled)

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape [n_samples, n_features]

        Returns:
            Probability matrix of shape [n_samples, n_classes]
        """
        if self._svm is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        if self.normalize and self._scaler is not None:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        return self._svm.predict_proba(X_scaled)

    def evaluate(
        self,
        X: NDArray[np.float32],
        y: list[SliceLabel] | NDArray,
    ) -> ClassificationResult:
        """Evaluate the classifier on a test set.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            ClassificationResult with metrics
        """
        # Encode labels
        if isinstance(y, list) and isinstance(y[0], SliceLabel):
            if self._label_encoder is None:
                raise RuntimeError("Classifier not fitted")
            y_encoded = self._label_encoder.transform([str(label) for label in y])
        else:
            y_encoded = np.array(y)

        predictions = self.predict(X)
        accuracy = accuracy_score(y_encoded, predictions)

        # Compute AUC for binary classification
        auc = None
        n_classes = len(np.unique(y_encoded))
        if n_classes == 2:
            proba = self.predict_proba(X)[:, 1]
            auc = roc_auc_score(y_encoded, proba)

        return ClassificationResult(
            accuracy=float(accuracy),
            auc=float(auc) if auc is not None else None,
            predictions=predictions,
            true_labels=y_encoded,
            kernel_type=self.kernel,
        )

    def cross_validate(
        self,
        X: NDArray[np.float32],
        y: list[SliceLabel] | NDArray,
        cv: int = 5,
    ) -> dict:
        """Perform cross-validation.

        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds

        Returns:
            Dictionary with mean and std of accuracy
        """
        # Encode labels
        if isinstance(y, list) and isinstance(y[0], SliceLabel):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform([str(label) for label in y])
        else:
            y_encoded = np.array(y)

        # Normalize
        if self.normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        # Cross-validate
        svm = self._create_svm()
        scores = cross_val_score(svm, X_scaled, y_encoded, cv=cv)

        return {
            "accuracy_mean": float(np.mean(scores)),
            "accuracy_std": float(np.std(scores)),
            "kernel_type": self.kernel,
            "cv_folds": cv,
        }

    def learning_curve(
        self,
        X: NDArray[np.float32],
        y: list[SliceLabel] | NDArray,
        train_sizes: Optional[NDArray] = None,
        cv: int = 5,
    ) -> LearningCurveResult:
        """Compute learning curve.

        Args:
            X: Feature matrix
            y: Labels
            train_sizes: Relative training set sizes to use
            cv: Number of CV folds

        Returns:
            LearningCurveResult with scores at each training size
        """
        if train_sizes is None:
            train_sizes = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

        # Encode labels
        if isinstance(y, list) and isinstance(y[0], SliceLabel):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform([str(label) for label in y])
        else:
            y_encoded = np.array(y)

        # Normalize
        if self.normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        svm = self._create_svm()

        train_sizes_abs, train_scores, test_scores = learning_curve(
            svm,
            X_scaled,
            y_encoded,
            train_sizes=train_sizes,
            cv=cv,
            shuffle=True,
            random_state=self.random_state,
        )

        return LearningCurveResult(
            train_sizes=train_sizes_abs,
            train_scores_mean=np.mean(train_scores, axis=1),
            train_scores_std=np.std(train_scores, axis=1),
            test_scores_mean=np.mean(test_scores, axis=1),
            test_scores_std=np.std(test_scores, axis=1),
            kernel_type=self.kernel,
        )


def train_classical_baseline(
    feature_matrix: WLFeatureMatrix,
    kernel: Literal["linear", "rbf"] = "rbf",
    random_state: int = 42,
) -> tuple[ClassicalKernelClassifier, dict]:
    """Train a classical kernel baseline on WL features.

    Args:
        feature_matrix: WL feature matrix from graph embeddings
        kernel: Kernel type
        random_state: Random seed

    Returns:
        Tuple of (fitted classifier, cross-validation results)
    """
    X = feature_matrix.to_matrix()
    y = feature_matrix.slice_labels

    classifier = ClassicalKernelClassifier(
        kernel=kernel,
        random_state=random_state,
    )

    # Cross-validate
    cv_results = classifier.cross_validate(X, y)

    # Fit on full data
    classifier.fit(X, y)

    return classifier, cv_results
