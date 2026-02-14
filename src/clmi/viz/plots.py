"""Matplotlib plotting helpers for CLMI experiment reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clmi.utils.io import ensure_dir


def plot_nc_forgetting_scatter(summary: pd.DataFrame, output_path: Path) -> None:
    """Scatter NC_global vs forgettingA."""
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(7, 5))
    for overlap, group in summary.groupby("overlap"):
        ax.scatter(
            group["NC_global"],
            group["forgettingA"],
            label=f"overlap={overlap}",
            alpha=0.75,
        )

    ax.set_xlabel("NC_global")
    ax.set_ylabel("forgettingA")
    ax.set_title("NC_global vs Forgetting")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_nc_layer_heatmap(layer_df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap of average NC_l by overlap."""
    ensure_dir(output_path.parent)
    pivot = layer_df.pivot_table(
        index="overlap",
        columns="layer",
        values="NC_l",
        aggfunc="mean",
        fill_value=0.0,
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(pivot.index)), labels=[str(x) for x in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[str(x) for x in pivot.columns])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Overlap")
    ax.set_title("Mean layer-wise NC by overlap")
    fig.colorbar(im, ax=ax, label="NC_l")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_hysteresis_curves(
    curves_by_overlap: dict[float, list[list[float]]],
    output_dir: Path,
) -> None:
    """Plot A->B->A retraining hysteresis curves by overlap."""
    ensure_dir(output_dir)

    for overlap, curves in curves_by_overlap.items():
        if not curves:
            continue

        max_len = max(len(curve) for curve in curves)
        arr = np.full((len(curves), max_len), np.nan, dtype=float)
        for i, curve in enumerate(curves):
            arr[i, : len(curve)] = curve

        mean_curve = np.nanmean(arr, axis=0)
        std_curve = np.nanstd(arr, axis=0)
        steps = np.arange(len(mean_curve))

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(steps, mean_curve, color="#1f77b4", label="mean Acc_A")
        ax.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color="#1f77b4",
            alpha=0.2,
            label="std",
        )
        ax.set_title(f"Hysteresis A->B->A (overlap={overlap})")
        ax.set_xlabel("Retraining step")
        ax.set_ylabel("Acc_A")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"hysteresis_overlap_{overlap}.png", dpi=150)
        plt.close(fig)


def plot_kernel_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    """Visualize task-task kernel matrix."""
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_mitigation_bars(summary: pd.DataFrame, output_path: Path) -> None:
    """Compare mitigations by mean forgetting."""
    ensure_dir(output_path.parent)
    grouped = summary.groupby("mitigation")["forgettingA"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(grouped.index.tolist(), grouped.values, color="#2ca02c", alpha=0.8)
    ax.set_ylabel("Mean forgettingA")
    ax.set_title("Mitigation comparison")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_full_loop(
    full_loop_dfs: list[pd.DataFrame],
    output_path: Path,
    title: str = "Full A→B→A loop",
) -> None:
    """Plot Acc_A and Acc_B trajectories across all 3 phases.

    Each DataFrame should contain columns 'acc_a' and 'acc_b' indexed by step.
    """
    ensure_dir(output_path.parent)

    # Pad all curves to the same length and average
    max_len = max(len(df) for df in full_loop_dfs)
    acc_a_arr = np.full((len(full_loop_dfs), max_len), np.nan)
    acc_b_arr = np.full((len(full_loop_dfs), max_len), np.nan)
    for i, df in enumerate(full_loop_dfs):
        a_vals = df["acc_a"].astype(float).values
        b_vals = df["acc_b"].astype(float).values
        acc_a_arr[i, : len(a_vals)] = a_vals
        acc_b_arr[i, : len(b_vals)] = b_vals

    steps = np.arange(max_len)
    mean_a = np.nanmean(acc_a_arr, axis=0)
    std_a = np.nanstd(acc_a_arr, axis=0)
    mean_b = np.nanmean(acc_b_arr, axis=0)
    std_b = np.nanstd(acc_b_arr, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, mean_a, color="#1f77b4", label="Acc_A (mean)")
    ax.fill_between(steps, mean_a - std_a, mean_a + std_a, color="#1f77b4", alpha=0.15)
    ax.plot(steps, mean_b, color="#d62728", label="Acc_B (mean)")
    ax.fill_between(steps, mean_b - std_b, mean_b + std_b, color="#d62728", alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Eval checkpoint (across phases A→B→A2)")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scatter_with_ci(
    summary: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    hue_col: str = "overlap",
    title: str | None = None,
) -> None:
    """Scatter plot with per-group mean ± std error bars."""
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(7, 5))
    groups = summary.groupby(hue_col) if hue_col in summary.columns else [(None, summary)]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # type: ignore[attr-defined]

    for idx, (label, group) in enumerate(groups):
        color = colors[idx % len(colors)]
        # Individual points (light)
        ax.scatter(group[x_col], group[y_col], color=color, alpha=0.3, s=20)
        # Group mean + std bars
        xm = group[x_col].mean()
        ym = group[y_col].mean()
        xs = group[x_col].std()
        ys = group[y_col].std()
        ax.errorbar(
            xm, ym, xerr=xs, yerr=ys,
            fmt="o", color=color, markersize=8, capsize=4,
            label=f"{hue_col}={label}" if label is not None else None,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{x_col} vs {y_col}")
    ax.grid(alpha=0.2)
    if len(list(groups)) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correlation_matrix(
    summary: pd.DataFrame,
    columns: list[str],
    output_path: Path,
    method: str = "spearman",
    title: str = "Metric correlations",
) -> None:
    """Heatmap of pairwise correlations among selected numerical columns."""
    ensure_dir(output_path.parent)

    # Keep only columns that exist and are numeric
    valid = [c for c in columns if c in summary.columns and np.issubdtype(summary[c].dtype, np.number)]
    if not valid:
        return

    corr = summary[valid].corr(method=method)

    fig, ax = plt.subplots(figsize=(max(6, len(valid) * 0.8), max(5, len(valid) * 0.7)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(valid)), labels=valid, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(valid)), labels=valid, fontsize=7)
    # Annotate cells
    for i in range(len(valid)):
        for j in range(len(valid)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(val) > 0.5 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=f"{method} r")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_weight_distance_bars(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bar chart of mean weight-space L2 distances per phase transition."""
    ensure_dir(output_path.parent)
    dist_cols = [c for c in summary.columns if c.startswith("wdist_") and c.endswith("_l2")]
    if not dist_cols:
        return

    means = summary[dist_cols].mean()
    stds = summary[dist_cols].std()
    labels = [c.replace("wdist_", "").replace("_l2", "") for c in dist_cols]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means.values, yerr=stds.values, capsize=4, color="#ff7f0e", alpha=0.8)
    ax.set_xticks(x, labels=labels, rotation=15, ha="right")
    ax.set_ylabel("L2 weight distance")
    ax.set_title("Weight-space distance per phase transition")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
