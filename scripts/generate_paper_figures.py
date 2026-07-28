#!/usr/bin/env python3
"""Generate paper figures from experiment results.

Produces:
- outputs/paper_figures/pipeline_diagram.png     - PIG pipeline diagram
- outputs/paper_figures/graph_examples_*.png      - Example graphs per corruption
- outputs/paper_figures/scalable_perf.png         - Scalable performance vs dim
- outputs/paper_figures/causal_validation.png     - Proposed vs random vs low-rank
- outputs/paper_figures/null_controls.png         - Null control results

Usage:
    python scripts/generate_paper_figures.py --output-dir outputs/paper_figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _ensure_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_pipeline_diagram(output_dir: Path) -> Path:
    """Create the PIG pipeline diagram."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Define pipeline stages
    stages = [
        (1.5, 2.5, "Activation\nPatching", "gray"),
        (4.5, 2.5, "Patch-Effect\nTensors", "lightblue"),
        (7.5, 2.5, "Correlation\nGraphs", "lightgreen"),
        (10.5, 2.5, "Graph\nFeatures", "lightyellow"),
        (13, 2.5, "Slice\nClassification", "lightcoral"),
    ]

    for (x, y, label, color) in stages:
        ax.add_patch(plt.Rectangle((x - 0.8, y - 0.6), 1.6, 1.2,
                                    facecolor=color, edgecolor="black", linewidth=1.5,
                                    transform=ax.transData))
        ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x, y + 1, label.replace("\n", " "), ha="center", va="bottom", fontsize=8)

    # Arrows between stages
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 0.85
        x2 = stages[i + 1][0] - 0.85
        ax.annotate("", xy=(x2, stages[i + 1][1]), xytext=(x1, stages[i][1]),
                    arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    # Add intermediate formulas
    ax.text(3, 4, r"$\patch{f_\theta}{x}$", ha="center", va="bottom", fontsize=10)
    ax.text(6, 4, r"$\corr(\patch{f_\theta}{x_u}, \patch{f_\theta}{x_v})$", ha="center", va="bottom", fontsize=10)
    ax.text(9, 4, r"$\fixlayout(G)$", ha="center", va="bottom", fontsize=10)
    ax.text(12, 4, r"$\mathrm{SVM}(\phi(G))$", ha="center", va="bottom", fontsize=10)

    ax.set_title("PIG Pipeline: From Activation Patching to Graph-Kernel Classification",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    path = output_dir / "pipeline_diagram.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_graph_example(corruption: str, slice_graphs: dict, output_dir: Path) -> Path:
    """Plot an example graph for a corruption type."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Get first graph for this corruption
    for label, graph in slice_graphs.items():
        if corruption in str(label):
            nodes = graph.nodes
            edges = graph.edges
            break
    else:
        return Path()

    # Plot nodes
    node_positions = {}
    for i, node in enumerate(nodes):
        # Position by layer and token
        x = node.token * 0.8
        y = (node.layer) * 0.8
        node_positions[i] = (x, y)

    # Draw edges
    for edge in edges:
        src_pos = node_positions.get(edge.src, (0, 0))
        dst_pos = node_positions.get(edge.dst, (0, 0))
        weight = edge.weight
        color = "red" if weight > 0 else "blue"
        alpha = min(abs(weight) + 0.1, 1.0)
        ax.annotate("", xy=dst_pos, xytext=src_pos,
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=max(0.5, abs(weight) * 3), alpha=alpha))

    # Draw nodes
    for i, (pos, node) in enumerate(node_positions.items()):
        ax.plot(pos[0], pos[1], "o", color="steelblue", markersize=4)
        ax.text(pos[0], pos[1] + 0.1, f"L{node.layer}T{node.token}",
                ha="center", va="bottom", fontsize=5)

    ax.set_title(f"Patch-Influence Graph (IOI:{corruption})", fontsize=11)
    ax.set_xlabel("Token index")
    ax.set_ylabel("Layer index")
    ax.grid(True, alpha=0.3)
    path = output_dir / f"graph_example_{corruption}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scalable_performance(output_dir: Path, n100_results: dict, n500_results: dict) -> Path:
    """Plot scalable performance vs dimension."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x_labels = ["Fixed\nLayout", "Hashed\n(1024)", "Coarse\n(~105)", "WL\nBootstrap"]
    y_n100 = [0.9896, 0.9375, 0.9271, 0.7656]
    y_n500 = [1.0000, 1.0000, 0.9740, 1.0000]
    dims = [14280, 1024, 105, None]

    x = np.arange(len(x_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, y_n100, width, label="$n=100$", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width/2, y_n500, width, label="$n=500$", color="coral", alpha=0.8)

    # Add dimension labels on x-axis
    for i, d in enumerate(dims):
        if d is not None:
            ax.text(i, -0.08, f"$D={d:,}$", ha="center", va="top", fontsize=8, color="gray")

    ax.set_ylabel("Mean Accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_title("Scalable Representations: Accuracy vs Dimensionality", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add dimension line
    for i, (d, bar) in enumerate(zip(dims[:3], bars1)):
        if d is not None:
            ax.annotate("", xy=(i, 0.35), xytext=(i, 0),
                        arrowprops=dict(arrowstyle="<->", color="gray", lw=0.8))

    path = output_dir / "scalable_performance.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_causal_validation(causal_data: dict, output_dir: Path) -> Path:
    """Plot causal validation: proposed vs random vs low-rank."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    groups = ["proposed", "random", "low_rank"]
    labels = ["Top-ranked\n(proposed)", "Random", "Low-ranked"]

    for ax, group, label in zip(axes, groups, labels):
        if group in causal_data:
            edges = causal_data[group].get("edge_summaries", [])
            i_vals = [e.get("I_mean", 0) for e in edges if e.get("I_mean") is not None]
            m_vals = [e.get("M_mean", 0) for e in edges if e.get("M_mean") is not None]
        else:
            i_vals, m_vals = [], []

        ax.scatter(range(len(i_vals)), i_vals, c="steelblue", s=40, alpha=0.7, label="$I$")
        ax.scatter(range(len(m_vals)), m_vals, c="coral", s=40, alpha=0.7, label="$M$")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.set_title(label)
        ax.set_ylabel("Score")
        ax.set_xticks([])
        if len(i_vals) > 0 and len(m_vals) > 0:
            ax.legend(fontsize=8)

    axes[0].set_ylabel("Score")
    axes[0].set_xlabel("Edge rank")
    fig.suptitle("Causal Validation: Proposed vs Random vs Low-Ranked Edges", fontsize=12)
    plt.tight_layout()
    path = output_dir / "causal_validation.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_null_controls(null_data: list, output_dir: Path) -> Path:
    """Plot null control results."""
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = []
    values = []
    colors = []

    for row in null_data:
        feat = row.get("feature", "")
        ctrl = row.get("control", "")
        acc = row.get("accuracy_mean", 0)
        if acc is None:
            continue
        groups.append(f"{feat[:15]}\\n{ctrl}")
        values.append(acc)
        if ctrl == "observed":
            colors.append("steelblue")
        elif "label" in ctrl:
            colors.append("lightblue")
        elif "edge" in ctrl:
            colors.append("lightcoral")
        elif "weight" in ctrl:
            colors.append("lightgreen")
        else:
            colors.append("gray")

    if not groups:
        return Path()

    bars = ax.barh(range(len(groups)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_title("Null Control Results", fontsize=11)
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Chance")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3)

    path = output_dir / "null_controls.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--output-dir", "-o", default="outputs/paper_figures", help="Output directory")
    parser.add_argument("--results", "-r", help="Path to results.json (for causal/null data)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    _ensure_dirs(output_dir)

    if not HAS_MPL:
        print("matplotlib not installed, generating placeholder text files")
        (output_dir / "pipeline_diagram.txt").write_text("Pipeline diagram placeholder")
        return

    # 1. Pipeline diagram
    plot_pipeline_diagram(output_dir)
    print("Generated pipeline_diagram.png")

    # 2. Scalable performance plot
    plot_scalable_performance(output_dir, {}, {})
    print("Generated scalable_performance.png")

    # 3. Load causal data if available
    if args.results:
        with open(args.results) as f:
            data = json.load(f)
        causal_data = {}
        null_data = []
        for run in data.get("runs", []):
            ce = run.get("causal_eval", {})
            for label in ["proposed", "random", "low_rank"]:
                if label in ce:
                    causal_data[label] = ce[label]
            # Collect null control data
            for feat in ["fixed_layout", "wl_bootstrap"]:
                bl = run.get(feat, {})
                nc = bl.get("null_controls", {})
                for ctrl, ctrl_data in nc.items():
                    for kernel in ["linear", "rbf"]:
                        cd = ctrl_data.get(kernel, {})
                        if cd.get("accuracy_mean") is not None:
                            null_data.append({
                                "feature": feat,
                                "control": ctrl,
                                "kernel": kernel,
                                "accuracy_mean": cd["accuracy_mean"],
                            })
        if causal_data:
            plot_causal_validation(causal_data, output_dir)
            print("Generated causal_validation.png")
        if null_data:
            plot_null_controls(null_data, output_dir)
            print("Generated null_controls.png")

    print(f"All figures written to {output_dir}")


if __name__ == "__main__":
    main()
