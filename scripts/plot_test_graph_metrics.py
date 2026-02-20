"""Quick visualization for test_graph_metrics MVP outputs.

Reads:
- outputs/test_graphs_metrics/results.npz
- outputs/test_graphs_metrics/summary.json

Writes:
- outputs/test_graphs_metrics/quicklook.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _annotate_heatmap(ax: plt.Axes, mat: np.ndarray) -> None:
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            txt = f"{mat[i, j]:+.3f}" if i != j else "0"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")


def plot_quicklook(input_dir: Path, output_path: Path) -> None:
    npz_path = input_dir / "results.npz"
    summary_path = input_dir / "summary.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    data = np.load(npz_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    R = data["R"]
    adj_sparse = data["adj_sparse"]
    G_A = data["G_A_dense"]
    G_B = data["G_B_dense"]

    off_diag = ~np.eye(G_A.shape[0], dtype=bool)
    wA = G_A[off_diag]
    wB = G_B[off_diag]

    rho = summary.get("split_half", {}).get("rho", float("nan"))
    pvalue = summary.get("split_half", {}).get("pvalue", float("nan"))
    n_edges = summary.get("split_half", {}).get("num_edges", int(wA.size))
    med = summary.get("mediation", {})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel 1: Restoration per node
    ax = axes[0, 0]
    nodes = np.arange(R.shape[0])
    ax.bar(nodes, R, color="#377eb8")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Restoration media por nodo (R_u)")
    ax.set_xlabel("Nodo")
    ax.set_ylabel("R_u")
    ax.set_xticks(nodes)

    # Panel 2: Sparse adjacency heatmap
    ax = axes[0, 1]
    vmax = float(np.max(np.abs(adj_sparse))) if adj_sparse.size else 1.0
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(adj_sparse, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("Grafo (adjacencia sparse)")
    ax.set_xlabel("Destino")
    ax.set_ylabel("Origen")
    ax.set_xticks(nodes)
    ax.set_yticks(nodes)
    _annotate_heatmap(ax, adj_sparse)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3: Split-half reproducibility
    ax = axes[1, 0]
    ax.scatter(wA, wB, c="#4daf4a", alpha=0.85)
    min_v = float(min(np.min(wA), np.min(wB))) if wA.size else -1.0
    max_v = float(max(np.max(wA), np.max(wB))) if wA.size else 1.0
    if min_v == max_v:
        min_v -= 0.1
        max_v += 0.1
    ax.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_title("Split-half: pesos de aristas (denso)")
    ax.set_xlabel("w_A")
    ax.set_ylabel("w_B")
    ax.text(
        0.02,
        0.98,
        f"rho={rho:+.3f}\np={pvalue:.3g}\nn_edges={n_edges}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    # Panel 4: Mediation snapshot
    ax = axes[1, 1]
    labels = ["R(u)", "R(v)", "R(u,v)"]
    vals = [med.get("R_u", np.nan), med.get("R_v", np.nan), med.get("R_uv", np.nan)]
    colors = ["#ff7f00", "#984ea3", "#e41a1c"]
    ax.bar(np.arange(3), vals, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_title("Mediación (simulada)")
    ax.set_ylabel("Restoration")
    ax.text(
        0.02,
        0.98,
        f"M(u->v)={med.get('M_uv', float('nan')):+.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    fig.suptitle(
        "Test Graph Metrics - Quicklook\n"
        f"source={summary.get('data_source', 'unknown')} | "
        f"n={summary.get('num_examples', '?')} | "
        f"nodes={summary.get('num_nodes', '?')}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quicklook for test_graph_metrics outputs")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/test_graphs_metrics"),
        help="Directory containing results.npz and summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/test_graphs_metrics/quicklook.png"),
        help="PNG output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_quicklook(args.input_dir, args.output)
    print(f"Quicklook guardado en: {args.output}")


if __name__ == "__main__":
    main()
