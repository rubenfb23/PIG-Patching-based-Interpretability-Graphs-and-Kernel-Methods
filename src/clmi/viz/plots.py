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
