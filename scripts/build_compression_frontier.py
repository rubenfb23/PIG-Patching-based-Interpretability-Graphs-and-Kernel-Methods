#!/usr/bin/env python3
"""Aggregate paper decision study results into compression frontier tables.

Reads all summary.csv files from outputs/paper_decision/*/summary.csv and
produces:
  - outputs/paper_compression_frontier.csv   (unified long-form)
  - outputs/paper_compression_frontier.md     (Markdown tables for the paper)
  - outputs/paper_figures/accuracy_dimensionality.png  (accuracy-dimensionality trade-off plot)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

REPR_META: dict[str, tuple[str, int | None, str]] = {
    "patch_effect_node_vector": ("Patch-effect (raw)", None, "gold"),
    "topk_node_identity": ("Top-k node", 10, "khaki"),
    "fixed_layout_weighted": ("Fixed layout (weighted)", None, "gold"),
    "fixed_layout_binary_topology": ("Fixed layout (binary)", None, "orange"),
    "fixed_layout_sign_topology": ("Fixed layout (signed)", None, "darkorange"),
    "wl_bootstrap": ("WL bootstrap", None, "steelblue"),
    "spectral_shape": ("Spectral shape", 96, "cornflowerblue"),
    "graphlet_shape": ("Graphlet shape", 38, "skyblue"),
    "hashed_fixed_sign": ("Hashed fixed (signed)", 1024, "green"),
    "hashed_fixed_weighted": ("Hashed fixed (weighted)", 1024, "lightgreen"),
    "coarse_fixed_count": ("Coarse count", 105, "teal"),
    "gnn_encoder_svm": ("GNN encoder", None, "purple"),
    "null_edge_shuffle": ("Null: edge-shuffle", None, "red"),
    "null_weight_shuffle": ("Null: weight-shuffle", None, "salmon"),
}

KEYS_OF_INTEREST = list(REPR_META.keys())

CSV_TO_KEY: dict[str, str] = {
    # Only linear kernel — RBF values are not used in the paper tables
    "wl_linear": "wl_bootstrap",
    "fixed_weighted_linear": "fixed_layout_weighted",
    "fixed_binary_linear": "fixed_layout_binary_topology",
    "fixed_sign_linear": "fixed_layout_sign_topology",
    "spectral_linear": "spectral_shape",
    "graphlet_linear": "graphlet_shape",
    "patch_effect_linear": "patch_effect_node_vector",
    "topk_node_linear": "topk_node_identity",
    "hashed_sign_linear": "hashed_fixed_sign",
    "hashed_weighted_linear": "hashed_fixed_weighted",
    "coarse_count_linear": "coarse_fixed_count",
    "gnn_encoder_svm_linear": "gnn_encoder_svm",
    "fixed_weighted_edge_shuffle_linear": "null_edge_shuffle",
    "fixed_weighted_weight_shuffle_linear": "null_weight_shuffle",
}

METRIC_KEY_FOR_REPR: dict[str, str] = {
    "null_edge_shuffle": "fixed_layout_weighted",
    "null_weight_shuffle": "fixed_layout_weighted",
}


def read_all_summaries() -> list[dict[str, Any]]:
    """Read summaries from the primary sweep directories only (skip old fragmented runs)."""
    base = Path("outputs/paper_decision")
    all_rows: list[dict[str, Any]] = []

    # Only include new sweeps (model_name_TIMESTAMP format) and n500 scalable
    allowed_dirs = {"gpt2_n500_res_k5_scalable_representations_20260430"}
    # Reverse sort makes newer timestamped runs win when deduplicating below.
    for csv_path in sorted(base.glob("*/summary.csv"), reverse=True):
        model_dir = csv_path.parent.name
        model = model_dir.split("_")[0]
        if model not in {"gpt2", "distilgpt2"}:
            continue
        # Include new sweeps (has 'T' timestamp) or n500 scalable
        if model_dir.startswith("gpt2_") and "T" in model_dir:
            pass  # include
        elif model_dir in allowed_dirs:
            pass  # include
        elif model_dir.startswith("distilgpt2_"):
            pass  # include
        else:
            continue  # skip old fragmented runs
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                n_val: int | None = None
                for part in row.get("run_key", "").split("_"):
                    if part.startswith("n") and part[1:].isdigit():
                        n_val = int(part[1:])
                        break
                row["model"] = model
                row["n"] = n_val
                row["_summary_dir"] = csv_path.parent
                all_rows.append(row)
    return all_rows


def _fixed_layout_dim(model: str) -> int | None:
    """Return fixed-layout dimension for the given model."""
    if model == "gpt2":
        return 14280
    elif model == "distilgpt2":
        return 3540
    return None


def _load_run_json(row: dict[str, Any]) -> dict[str, Any] | None:
    summary_dir = row.get("_summary_dir")
    run_key = row.get("run_key")
    if not summary_dir or not run_key:
        return None
    run_path = Path(summary_dir) / "runs" / f"{run_key}.json"
    if not run_path.exists():
        return None
    with open(run_path, encoding="utf-8") as fh:
        return json.load(fh)


def _dimension_from_run(row: dict[str, Any], internal_key: str) -> int | None:
    """Read feature dimensionality from the per-seed JSON whenever available."""
    run = _load_run_json(row)
    metric_key = METRIC_KEY_FOR_REPR.get(internal_key, internal_key)
    metric = (run or {}).get("paper_decision", {}).get("metrics", {}).get(metric_key, {})
    linear_metric = metric.get("linear", {})
    dim = linear_metric.get("num_features", linear_metric.get("embedding_dim"))
    if isinstance(dim, int):
        return dim

    base_dim = REPR_META.get(internal_key, (internal_key, None, "gray"))[1]
    if base_dim is not None:
        return base_dim
    if internal_key.startswith("fixed_layout") or internal_key.startswith("null_"):
        return _fixed_layout_dim(str(row.get("model", "")))
    return None


def _aggregate_dim(dims: list[int]) -> int | str | None:
    if not dims:
        return None
    unique_dims = sorted(set(dims))
    if len(unique_dims) == 1:
        return unique_dims[0]
    return f"{unique_dims[0]}--{unique_dims[-1]}"


def _format_dim(dim: int | str | None) -> str:
    if dim is None:
        return "---"
    if isinstance(dim, int):
        return f"{dim:,}"
    return dim


def _numeric_dim(dim: int | str | None) -> float | None:
    if isinstance(dim, int):
        return float(dim)
    if isinstance(dim, str) and "--" in dim:
        low, high = dim.split("--", maxsplit=1)
        if low.isdigit() and high.isdigit():
            return (float(low) + float(high)) / 2
    return None


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate across seeds. Deduplicates by (model, n, seed) keeping only the first
    occurrence per seed (so newer sweep results take priority over older runs)."""
    groups: dict[tuple, list[tuple[float, int | None]]] = {}
    seen_seeds: set[tuple] = set()
    for row in rows:
        model = row["model"]
        n = row["n"]
        seed = int(row["seed"])
        seed_key = (model, n, seed)
        if seed_key in seen_seeds:
            continue  # deduplicate: newer sweep rows take priority
        seen_seeds.add(seed_key)
        for csv_col, internal_key in CSV_TO_KEY.items():
            if internal_key not in KEYS_OF_INTEREST:
                continue
            acc_str = row.get(csv_col, "")
            if not acc_str:
                continue
            acc = float(acc_str)
            if not math.isfinite(acc):
                continue
            dim = _dimension_from_run(row, internal_key)
            groups.setdefault((model, n, internal_key), []).append((acc, dim))

    results = []
    for (model, n, key), values in sorted(groups.items()):
        accs = [acc for acc, _ in values]
        dims = [dim for _, dim in values if dim is not None]
        mean_val = sum(accs) / len(accs)
        std_val = (sum((a - mean_val) ** 2 for a in accs) / len(accs)) ** 0.5
        base_meta = REPR_META.get(key, (key, None, "gray"))
        dim = _aggregate_dim(dims)
        results.append({
            "model": model,
            "n": n,
            "representation": base_meta[0],
            "key": key,
            "accuracy_mean": round(mean_val, 4),
            "accuracy_std": round(std_val, 4),
            "dim": dim,
            "n_seeds": len(accs),
        })
    return results


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["model", "n", "representation", "key", "accuracy_mean",
                  "accuracy_std", "dim", "n_seeds"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def write_markdown(results: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    for (model, n), group in _group_by_model_n(results):
        lines.append(f"## {model} (n={n})")
        lines.append("")
        lines.append("| Representation | Accuracy | Std | Dim |")
        lines.append("|---|---|---|---|")
        for r in sorted(group, key=lambda x: -x["accuracy_mean"]):
            dim_str = _format_dim(r["dim"])
            lines.append(f"| {r['representation']} | {r['accuracy_mean']:.4f} | "
                         f"{r['accuracy_std']:.4f} | {dim_str} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _group_by_model_n(results: list[dict[str, Any]]) -> list[tuple[tuple[str, int], list[dict]]]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in results:
        groups.setdefault((r["model"], r["n"]), []).append(r)
    return [(k, sorted(v, key=lambda x: -x["accuracy_mean"])) for k, v in groups.items()]


def plot_pareto(results: list[dict[str, Any]], path: Path) -> None:
    if not HAS_MPL:
        (path.parent / "accuracy_dimensionality.txt").write_text(
            "matplotlib not available, skipping accuracy-dimensionality plot"
        )
        return

    fig, ax = plt.subplots(figsize=(11, 6.5))

    models = sorted(set(r["model"] for r in results))
    colors = {"gpt2": "steelblue", "distilgpt2": "green", "toy_transformer": "gold"}
    markers = {"gpt2": "o", "distilgpt2": "s", "toy_transformer": "^"}

    for model in models:
        model_results = [r for r in results if r["model"] == model]
        x, y, plotted = [], [], []
        for r in model_results:
            dim = _numeric_dim(r["dim"])
            if dim is not None and dim > 0:
                x.append(math.log10(dim))
                y.append(r["accuracy_mean"])
                plotted.append(r)
        ax.scatter(x, y, c=colors.get(model, "gray"),
                   marker=markers.get(model, "o"), s=80,
                   label=f"{model}", alpha=0.7, edgecolors="black", linewidth=0.5)
        label_offsets = {
            "WL bootstrap": ("WL", (6, 6)),
            "Fixed layout (signed)": ("Fixed sign", (6, -10)),
            "Hashed fixed (signed)": ("Hashed sign", (6, -10)),
            "Coarse count": ("Coarse", (6, 7)),
            "Spectral shape": ("Spectral", (6, -12)),
            "Graphlet shape": ("Graphlet", (6, 7)),
        }
        for xi, yi, r in zip(x, y, plotted):
            if not (r["model"] == "gpt2" and r["n"] == 100):
                continue
            name = str(r["representation"])
            if name not in label_offsets:
                continue
            label, (dx, dy) = label_offsets[name]
            ax.annotate(label, (xi, yi),
                        textcoords="offset points", xytext=(dx, dy), fontsize=7.5,
                        color=colors.get(model, "gray"))

    # Connect by n for GPT-2
    gpt2 = [r for r in results if r["model"] == "gpt2"]
    for n in sorted(set(r["n"] for r in gpt2)):
        n_data = [r for r in gpt2 if r["n"] == n and (_numeric_dim(r["dim"]) or 0) > 0]
        if len(n_data) >= 2:
            gpt2_n = sorted(n_data, key=lambda x: -math.log10(_numeric_dim(x["dim"]) or 1))
            ax.plot([math.log10(_numeric_dim(r["dim"]) or 1) for r in gpt2_n],
                    [r["accuracy_mean"] for r in gpt2_n],
                    "--", c="gray", alpha=0.3, linewidth=0.8)

    ax.set_xlabel(r"log$_{10}$(Feature Dimension)", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title("Accuracy-Dimensionality Trade-off: Signal Compression", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=-1)

    plt.tight_layout()
    fig.savefig(str(path), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = Path("outputs")
    base.mkdir(exist_ok=True)

    print("Reading all summaries...")
    rows = read_all_summaries()
    sources = set(r["model"] + str(r["n"]) for r in rows)
    print(f"  Read {len(rows)} rows from {len(sources)} sources: {sources}")

    print("Aggregating across seeds...")
    results = aggregate(rows)
    print(f"  {len(results)} aggregated entries")

    csv_path = base / "paper_compression_frontier.csv"
    write_csv(results, csv_path)
    print(f"  Written {csv_path}")

    md_path = base / "paper_compression_frontier.md"
    write_markdown(results, md_path)
    print(f"  Written {md_path}")

    fig_dir = base / "paper_figures"
    fig_dir.mkdir(exist_ok=True)
    pareto_path = fig_dir / "accuracy_dimensionality.png"
    plot_pareto(results, pareto_path)
    print(f"  Written {pareto_path}")

    # Print summary for the user
    for (model, n), group in _group_by_model_n(results):
        print(f"\n=== {model} n={n} ===")
        for r in sorted(group, key=lambda x: -x["accuracy_mean"]):
            dim_str = _format_dim(r["dim"])
            print(f"  {r['representation']:30s} {r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}  D={dim_str}")


if __name__ == "__main__":
    main()
