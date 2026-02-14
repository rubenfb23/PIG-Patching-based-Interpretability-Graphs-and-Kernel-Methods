"""Statistical analysis utilities for CLMI experiment results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats as sp_stats
except ImportError:  # pragma: no cover
    sp_stats = None  # type: ignore[assignment]


def _require_scipy() -> None:
    if sp_stats is None:
        raise RuntimeError("scipy is required for statistical analysis. Install `scipy`.")


def pearson_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    bootstrap_iters: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Pearson r with bootstrap confidence interval and p-value."""
    _require_scipy()
    r, p = sp_stats.pearsonr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rs: list[float] = []
    for _ in range(bootstrap_iters):
        idx = rng.choice(n, size=n, replace=True)
        boot_r, _ = sp_stats.pearsonr(x[idx], y[idx])
        boot_rs.append(float(boot_r))
    boot_arr = np.asarray(boot_rs)
    lo = float(np.percentile(boot_arr, 100 * alpha / 2))
    hi = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
    return {"r": float(r), "p": float(p), "ci_lo": lo, "ci_hi": hi}


def spearman_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    bootstrap_iters: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Spearman rho with bootstrap confidence interval and p-value."""
    _require_scipy()
    rho, p = sp_stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rhos: list[float] = []
    for _ in range(bootstrap_iters):
        idx = rng.choice(n, size=n, replace=True)
        boot_rho, _ = sp_stats.spearmanr(x[idx], y[idx])
        boot_rhos.append(float(boot_rho))
    boot_arr = np.asarray(boot_rhos)
    lo = float(np.percentile(boot_arr, 100 * alpha / 2))
    hi = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
    return {"rho": float(rho), "p": float(p), "ci_lo": lo, "ci_hi": hi}


def paired_permutation_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Two-sided paired permutation test on mean difference."""
    diff = values_a - values_b
    observed = float(np.abs(np.mean(diff)))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diff))
        perm_diff = float(np.abs(np.mean(diff * signs)))
        if perm_diff >= observed:
            count += 1
    p = (count + 1) / (n_permutations + 1)
    return {"mean_diff": float(np.mean(diff)), "p_perm": float(p)}


def bootstrap_mean_ci(
    values: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        means.append(float(np.mean(values[idx])))
    arr = np.asarray(means)
    return {
        "mean": float(np.mean(values)),
        "ci_lo": float(np.percentile(arr, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def compute_correlation_table(
    summary: pd.DataFrame,
    predictor_cols: list[str],
    target_col: str = "forgettingA",
    seed: int = 0,
) -> pd.DataFrame:
    """Compute Pearson + Spearman correlations between predictors and a target."""
    _require_scipy()
    rows: list[dict[str, Any]] = []
    y = summary[target_col].to_numpy(dtype=float)
    valid = np.isfinite(y)

    for col in predictor_cols:
        if col not in summary.columns:
            continue
        x = summary[col].to_numpy(dtype=float)
        mask = valid & np.isfinite(x)
        if mask.sum() < 4:
            continue
        xm, ym = x[mask], y[mask]
        pear = pearson_with_ci(xm, ym, seed=seed)
        spear = spearman_with_ci(xm, ym, seed=seed)
        rows.append(
            {
                "predictor": col,
                "pearson_r": pear["r"],
                "pearson_p": pear["p"],
                "pearson_ci_lo": pear["ci_lo"],
                "pearson_ci_hi": pear["ci_hi"],
                "spearman_rho": spear["rho"],
                "spearman_p": spear["p"],
                "spearman_ci_lo": spear["ci_lo"],
                "spearman_ci_hi": spear["ci_hi"],
            }
        )
    return pd.DataFrame(rows)
