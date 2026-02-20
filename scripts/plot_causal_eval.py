#!/usr/bin/env python3
"""Generate quicklook figure for causal evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot-causal-eval",
        description="Plot quicklook_causal.png from causal_eval JSON/NPZ",
    )
    parser.add_argument(
        "--json-path",
        default="outputs/causal_eval/causal_eval.json",
        help="Path to causal_eval.json",
    )
    parser.add_argument(
        "--npz-path",
        default="outputs/causal_eval/causal_eval.npz",
        help="Path to causal_eval.npz",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/causal_eval/quicklook_causal.png",
        help="Output PNG path",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    json_path = Path(args.json_path)
    npz_path = Path(args.npz_path)
    output_path = Path(args.output_path)

    if not json_path.exists():
        print(f"[FAIL] Missing JSON: {json_path}")
        return 1
    if not npz_path.exists():
        print(f"[FAIL] Missing NPZ: {npz_path}")
        return 1

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    arrays = np.load(npz_path)
    edge_ids = arrays["edge_ids"].astype(str)
    src_labels = arrays["src_labels"].astype(str)
    dst_labels = arrays["dst_labels"].astype(str)

    i_mean = arrays["I_mean"].astype(float)
    i_low = arrays["I_ci_low"].astype(float)
    i_high = arrays["I_ci_high"].astype(float)

    r_u = arrays["R_u_mean"].astype(float)
    r_v = arrays["R_v_mean"].astype(float)
    r_uv = arrays["R_uv_mean"].astype(float)
    r_u_clamp = arrays["R_u_clamp_v_mean"].astype(float)

    necessity = arrays["necessity_mean"].astype(float)
    necessity_low = arrays["necessity_ci_low"].astype(float)
    necessity_high = arrays["necessity_ci_high"].astype(float)

    n_edges = len(edge_ids)
    if n_edges == 0:
        print("[FAIL] Empty edge set in NPZ")
        return 1

    unique_src = list(dict.fromkeys(src_labels.tolist()))
    unique_dst = list(dict.fromkeys(dst_labels.tolist()))
    src_index = {label: idx for idx, label in enumerate(unique_src)}
    dst_index = {label: idx for idx, label in enumerate(unique_dst)}
    heatmap = np.full((len(unique_src), len(unique_dst)), np.nan, dtype=np.float64)
    for idx in range(n_edges):
        heatmap[src_index[src_labels[idx]], dst_index[dst_labels[idx]]] = i_mean[idx]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    # Panel 1: I(u->v) heatmap
    ax = axes[0, 0]
    im = ax.imshow(np.nan_to_num(heatmap, nan=0.0), aspect="auto", cmap="viridis")
    ax.set_title("Level A: I(u->v)")
    ax.set_xlabel("v (destination)")
    ax.set_ylabel("u (source)")
    ax.set_xticks(range(len(unique_dst)))
    ax.set_xticklabels(unique_dst, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(unique_src)))
    ax.set_yticklabels(unique_src, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: R(u), R(v), R(u,v)
    ax = axes[0, 1]
    x = np.arange(n_edges)
    w = 0.26
    ax.bar(x - w, r_u, width=w, label="R(u)")
    ax.bar(x, r_v, width=w, label="R(v)")
    ax.bar(x + w, r_uv, width=w, label="R(u,v)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Level B: Restoration")
    ax.set_xlabel("Edge")
    ax.set_ylabel("Restoration fraction")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)
    ax.legend(fontsize=8)

    # Panel 3: clamp necessity view
    ax = axes[1, 0]
    w = 0.38
    ax.bar(x - w / 2.0, r_u, width=w, label="R(u)")
    ax.bar(x + w / 2.0, r_u_clamp, width=w, label="R(u, clamp_v)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Level C: Path Blocking")
    ax.set_xlabel("Edge")
    ax.set_ylabel("Restoration fraction")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)
    ax.legend(fontsize=8)

    # Panel 4: stability / CI summary
    ax = axes[1, 1]
    i_err = np.vstack([i_mean - i_low, i_high - i_mean])
    n_err = np.vstack([necessity - necessity_low, necessity_high - necessity])
    ax.errorbar(
        x - 0.08,
        i_mean,
        yerr=i_err,
        fmt="o",
        capsize=3,
        label="I mean ± CI",
    )
    ax.errorbar(
        x + 0.08,
        necessity,
        yerr=n_err,
        fmt="s",
        capsize=3,
        label="necessity mean ± CI",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Stability / CI by Edge")
    ax.set_xlabel("Edge")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)
    ax.legend(fontsize=8)

    model_name = payload.get("model_name", "unknown")
    n_examples = payload.get("run_metadata", {}).get("num_prompt_pairs_used", "?")
    fig.suptitle(f"Causal Quicklook ({model_name}, examples={n_examples})", fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved quicklook: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
