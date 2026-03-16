"""Unit tests for pig.kernels module."""

import numpy as np
import pytest

from pig.embeddings import WLEmbedding, WLFeatureMatrix
import pig.kernels as kernels_module
from pig.kernels import (
    ClassicalKernelClassifier,
    train_classical_baseline,
)
from pig.prompts import SliceLabel


@pytest.fixture
def sample_data():
    """Create sample classification data."""
    np.random.seed(42)

    # Create two classes with separable features
    n_per_class = 20
    n_features = 10

    # Class 0: centered around 0
    X0 = np.random.randn(n_per_class, n_features).astype(np.float32)

    # Class 1: centered around 2
    X1 = np.random.randn(n_per_class, n_features).astype(np.float32) + 2

    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    return X, y


@pytest.fixture
def sample_feature_matrix():
    """Create a sample WLFeatureMatrix."""
    np.random.seed(42)

    # Create embeddings for two slices
    slice_labels = [
        SliceLabel(task="ioi", corruption="name_swap"),
        SliceLabel(task="ioi", corruption="abba"),
    ]

    embeddings = []
    vocabulary = [f"f{i}" for i in range(20)]

    for label in slice_labels:
        features = {f"f{i}": np.random.randint(0, 10) for i in range(20)}
        emb = WLEmbedding(features=features, graph_id=str(label), depth=3)
        embeddings.append(emb)

    # Duplicate to have more samples
    for _ in range(9):
        for label in slice_labels:
            features = {f"f{i}": np.random.randint(0, 10) for i in range(20)}
            emb = WLEmbedding(features=features, graph_id=str(label), depth=3)
            embeddings.append(emb)

    all_labels = slice_labels * 10

    return WLFeatureMatrix(
        embeddings=embeddings,
        vocabulary=vocabulary,
        slice_labels=all_labels,
    )


class TestClassicalKernelClassifier:
    """Tests for ClassicalKernelClassifier."""

    def test_linear_kernel(self, sample_data):
        """Test linear kernel SVM."""
        X, y = sample_data

        clf = ClassicalKernelClassifier(kernel="linear", random_state=42)
        clf.fit(X, y)

        predictions = clf.predict(X)
        accuracy = np.mean(predictions == y)

        # Should achieve reasonable accuracy on separable data
        assert accuracy > 0.8

    def test_rbf_kernel(self, sample_data):
        """Test RBF kernel SVM."""
        X, y = sample_data

        clf = ClassicalKernelClassifier(kernel="rbf", random_state=42)
        clf.fit(X, y)

        predictions = clf.predict(X)
        accuracy = np.mean(predictions == y)

        assert accuracy > 0.8

    def test_predict_proba(self, sample_data):
        """Test probability predictions."""
        X, y = sample_data

        clf = ClassicalKernelClassifier(kernel="rbf", random_state=42)
        clf.fit(X, y)

        proba = clf.predict_proba(X)

        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_evaluate(self, sample_data):
        """Test evaluation metrics."""
        X, y = sample_data

        # Split into train/test
        train_idx = np.arange(0, 30)
        test_idx = np.arange(30, 40)

        clf = ClassicalKernelClassifier(kernel="rbf", random_state=42)
        clf.fit(X[train_idx], y[train_idx])

        result = clf.evaluate(X[test_idx], y[test_idx])

        assert 0 <= result.accuracy <= 1
        assert result.auc is not None
        assert 0 <= result.auc <= 1
        assert result.kernel_type == "rbf"

    def test_cross_validate(self, sample_data):
        """Test cross-validation."""
        X, y = sample_data

        clf = ClassicalKernelClassifier(kernel="linear", random_state=42)
        cv_result = clf.cross_validate(X, y, cv=3)

        assert "accuracy_mean" in cv_result
        assert "accuracy_std" in cv_result
        assert cv_result["accuracy_mean"] > 0.5  # Above chance

    def test_cross_validate_uses_pipeline_when_normalized(self, sample_data, monkeypatch):
        """Test CV uses a fold-local scaler instead of fitting on all data."""
        X, y = sample_data
        observed = {}

        def fake_cross_val_score(estimator, X_arg, y_arg, cv):
            observed["estimator"] = estimator
            observed["X"] = X_arg
            observed["y"] = y_arg
            observed["cv"] = cv
            return np.array([0.6, 0.7, 0.8], dtype=np.float64)

        monkeypatch.setattr(kernels_module, "cross_val_score", fake_cross_val_score)

        clf = ClassicalKernelClassifier(kernel="linear", random_state=42, normalize=True)
        cv_result = clf.cross_validate(X, y, cv=3)

        assert hasattr(observed["estimator"], "named_steps")
        assert list(observed["estimator"].named_steps) == ["scaler", "svm"]
        np.testing.assert_array_equal(observed["X"], X)
        np.testing.assert_array_equal(observed["y"], y)
        assert cv_result["cv_folds"] == 3

    def test_learning_curve_uses_pipeline_when_normalized(self, sample_data, monkeypatch):
        """Test learning-curve CV also avoids pre-fitting the scaler."""
        X, y = sample_data
        observed = {}

        def fake_learning_curve(estimator, X_arg, y_arg, train_sizes, cv, shuffle, random_state):
            observed["estimator"] = estimator
            observed["X"] = X_arg
            observed["y"] = y_arg
            observed["cv"] = cv
            return (
                np.array([5, 10], dtype=np.int32),
                np.array([[0.9, 0.8], [0.95, 0.9]], dtype=np.float64),
                np.array([[0.6, 0.7], [0.7, 0.75]], dtype=np.float64),
            )

        monkeypatch.setattr(kernels_module, "learning_curve", fake_learning_curve)

        clf = ClassicalKernelClassifier(kernel="linear", random_state=42, normalize=True)
        lc = clf.learning_curve(X, y, train_sizes=np.array([0.5, 1.0]), cv=3)

        assert hasattr(observed["estimator"], "named_steps")
        assert list(observed["estimator"].named_steps) == ["scaler", "svm"]
        np.testing.assert_array_equal(observed["X"], X)
        np.testing.assert_array_equal(observed["y"], y)
        assert lc.kernel_type == "linear"

    def test_cross_validate_caps_invalid_fold_count(self):
        """Test requested CV folds are reduced to the valid stratified maximum."""
        X = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [1.0, 1.0],
                [1.1, 1.1],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 0, 1, 1])

        clf = ClassicalKernelClassifier(kernel="linear", random_state=42)
        result = clf.cross_validate(X, y, cv=5)

        assert result["cv_folds"] == 2

    def test_learning_curve(self, sample_data):
        """Test learning curve computation."""
        X, y = sample_data

        clf = ClassicalKernelClassifier(kernel="linear", random_state=42)
        lc = clf.learning_curve(X, y, cv=3)

        assert len(lc.train_sizes) > 0
        assert len(lc.train_scores_mean) == len(lc.train_sizes)
        assert len(lc.test_scores_mean) == len(lc.train_sizes)

        # Training scores should generally increase with more data
        # (or stay high)
        assert lc.train_scores_mean[-1] >= 0.5

    def test_reproducibility(self, sample_data):
        """Test that same seed gives same results."""
        X, y = sample_data

        clf1 = ClassicalKernelClassifier(kernel="rbf", random_state=42)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)

        clf2 = ClassicalKernelClassifier(kernel="rbf", random_state=42)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)

    def test_with_slice_labels(self, sample_feature_matrix):
        """Test with SliceLabel objects."""
        X = sample_feature_matrix.to_matrix()
        y = sample_feature_matrix.slice_labels

        clf = ClassicalKernelClassifier(kernel="linear", random_state=42)
        clf.fit(X, y)

        predictions = clf.predict(X)
        assert len(predictions) == len(X)


class TestTrainClassicalBaseline:
    """Tests for train_classical_baseline function."""

    def test_basic(self, sample_feature_matrix):
        """Test basic training."""
        clf, cv_results = train_classical_baseline(
            sample_feature_matrix,
            kernel="linear",
            random_state=42,
        )

        assert clf is not None
        assert "accuracy_mean" in cv_results
        assert cv_results["kernel_type"] == "linear"

    def test_rbf(self, sample_feature_matrix):
        """Test with RBF kernel."""
        clf, cv_results = train_classical_baseline(
            sample_feature_matrix,
            kernel="rbf",
            random_state=42,
        )

        assert cv_results["kernel_type"] == "rbf"
