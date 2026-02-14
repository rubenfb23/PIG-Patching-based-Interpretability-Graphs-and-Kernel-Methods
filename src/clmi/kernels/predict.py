"""Kernel-ridge utilities to predict forgetting/interference."""

from __future__ import annotations

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


def fit_kernel_ridge_predictor(
    kernel_matrix: np.ndarray,
    targets: np.ndarray,
    alpha: float = 1.0,
    cv_folds: int = 5,
) -> dict[str, object]:
    """Fit a kernel-ridge model and report train + CV metrics."""
    if kernel_matrix.shape[0] != kernel_matrix.shape[1]:
        raise ValueError("kernel_matrix must be square.")
    if kernel_matrix.shape[0] != targets.shape[0]:
        raise ValueError("targets length must match kernel_matrix.")

    model = KernelRidge(alpha=alpha, kernel="precomputed")
    model.fit(kernel_matrix, targets)
    pred_train = model.predict(kernel_matrix)

    results: dict[str, object] = {
        "model": model,
        "train_rmse": float(np.sqrt(mean_squared_error(targets, pred_train))),
        "train_r2": float(r2_score(targets, pred_train)),
        "pred_train": pred_train,
    }

    if kernel_matrix.shape[0] >= max(3, cv_folds):
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
        cv_pred = np.zeros_like(targets, dtype=float)

        for train_idx, test_idx in kf.split(kernel_matrix):
            K_train = kernel_matrix[np.ix_(train_idx, train_idx)]
            K_test = kernel_matrix[np.ix_(test_idx, train_idx)]
            y_train = targets[train_idx]

            fold_model = KernelRidge(alpha=alpha, kernel="precomputed")
            fold_model.fit(K_train, y_train)
            cv_pred[test_idx] = fold_model.predict(K_test)

        results.update(
            {
                "cv_rmse": float(np.sqrt(mean_squared_error(targets, cv_pred))),
                "cv_r2": float(r2_score(targets, cv_pred)),
                "pred_cv": cv_pred,
            }
        )

    return results
