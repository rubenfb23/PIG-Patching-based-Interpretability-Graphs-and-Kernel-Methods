#!/usr/bin/env python3
"""Plot how representation dimensionality scales with analyzed model size.

The relevant scaling variable for patch-effect graph representations is the
number of analyzed internal nodes:

    N = layers * token positions * component types.

This script produces a log-log line plot comparing exact fixed-layout features
against compressed alternatives whose dimensionality is independent of N.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _scaling_rows(nodes: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for n in nodes:
        rows.append(
            {
                "nodes": int(n),
                "raw_patch_effect": int(n),
                "fixed_layout": int(n * (n - 1)),
                "hashed_fixed_layout": 1024,
                "coarse_count": 105,
                "spectral_shape": 96,
                "graphlet_shape": 38,
            }
        )
    return rows


def plot_compression_scaling(output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Includes the paper's concrete scales:
    # distilGPT-2 residual IOI: N=60, GPT-2 residual IOI: N=120,
    # 32 layers x 512 tokens x residual: N=16384,
    # 32 layers x 512 tokens x 3 component types: N=49152.
    nodes = np.array([60, 120, 240, 480, 960, 1920, 4096, 8192, 16384, 49152])
    rows = _scaling_rows(nodes)

    csv_path = output_dir / "compression_scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    series = [
        ("Fixed layout", "fixed_layout", "#1f4e79", "-", "o", 2.4),
        ("Raw patch effects", "raw_patch_effect", "#8a8a8a", "--", "s", 1.7),
        ("Hashed fixed layout", "hashed_fixed_layout", "#c44e52", "-", "^", 2.0),
        ("Coarse count", "coarse_count", "#55a868", "-", "D", 2.0),
        ("Spectral shape", "spectral_shape", "#8172b2", "-", "v", 1.8),
        ("Graphlet shape", "graphlet_shape", "#ccb974", "-", "P", 1.8),
    ]

    for label, key, color, linestyle, marker, linewidth in series:
        ax.plot(
            nodes,
            [row[key] for row in rows],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=linewidth,
            markersize=4.5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Analyzed internal nodes $N$")
    ax.set_ylabel("Feature dimension")
    ax.set_title("Representation Dimension vs. Analyzed Model Size")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    annotations = [
        (60, "DistilGPT-2\nIOI/res"),
        (120, "GPT-2\nIOI/res"),
        (16384, "32L x 512T\nres"),
        (49152, "32L x 512T\nres+att+mlp"),
    ]
    for x, text in annotations:
        y = x * (x - 1)
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=7.5,
            color="#333333",
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.6},
        )

    ax.text(
        0.985,
        0.035,
        "Fixed layout grows as $N(N-1)$; hashed/coarse/shape features stay bounded.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )

    fig.tight_layout()
    png_path = output_dir / "compression_scaling.png"
    pdf_path = output_dir / "compression_scaling.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_figures",
        help="Directory where the figure and CSV are written.",
    )
    args = parser.parse_args()
    paths = plot_compression_scaling(Path(args.output_dir))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
