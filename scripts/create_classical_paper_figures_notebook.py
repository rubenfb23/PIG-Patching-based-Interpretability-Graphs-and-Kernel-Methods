from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("notebooks/classical_publication_figures.ipynb")


def _source(text: str) -> list[str]:
    return dedent(text).lstrip("\n").splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Classical Publication Figures

            This notebook builds the paper-oriented figures for the current classical publication path:

            - `gpt2`
            - `node_types=res`
            - `k=5`
            - `num_examples_per_corruption=20`
            - `seeds={7,42,123}`

            It focuses on the story we decided to publish:

            1. The WL+SVM classifier has real but moderate auxiliary signal.
            2. The causal result is strong with the `clean_score > base_score` filter.
            3. That filter also introduces a large slice imbalance and materially strengthens the apparent causal result.

            The notebook exports ready-to-use figures to `outputs/paper_figures/classical_publication/`.
            """
        ),
        code_cell(
            r"""
            import csv
            import json
            from pathlib import Path
            from statistics import mean, stdev

            import matplotlib.pyplot as plt
            import numpy as np

            plt.style.use("seaborn-v0_8-whitegrid")
            plt.rcParams.update(
                {
                    "figure.dpi": 140,
                    "savefig.dpi": 200,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "axes.titleweight": "bold",
                    "axes.labelsize": 11,
                    "axes.titlesize": 13,
                    "legend.frameon": False,
                }
            )

            def find_repo_root(start: Path | None = None) -> Path:
                start = (start or Path.cwd()).resolve()
                for candidate in [start, *start.parents]:
                    if (candidate / "pyproject.toml").exists() and (candidate / "outputs" / "classical_publication").exists():
                        return candidate
                raise FileNotFoundError(
                    "Could not locate repo root containing pyproject.toml and outputs/classical_publication"
                )

            ROOT = find_repo_root()
            FILTERED_DIR = ROOT / "outputs" / "classical_publication" / "gpt2_multiseed_res_k5_n20"
            FILTERED_N12_DIR = ROOT / "outputs" / "classical_publication" / "gpt2_multiseed_res_k5_n12"
            NO_FILTER_DIR = ROOT / "outputs" / "classical_publication" / "gpt2_multiseed_res_k5_n20_no_filter"
            FIGURE_DIR = ROOT / "outputs" / "paper_figures" / "classical_publication"
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)

            COLORS = {
                "linear": "#0d3b66",
                "rbf": "#f4a261",
                "filtered": "#2a9d8f",
                "all_examples": "#e76f51",
                "abba": "#b22222",
                "name_swap": "#1f77b4",
            }
            SLICE_LABELS = {
                "ioi:abba": "ABBA",
                "ioi:name_swap": "Name swap",
            }

            def read_summary(path: Path) -> list[dict]:
                with path.open(newline="") as f:
                    rows = list(csv.DictReader(f))
                parsed = []
                for row in rows:
                    item = dict(row)
                    for key in [
                        "seed",
                        "k",
                        "num_examples_per_corruption",
                        "linear_accuracy_mean",
                        "linear_accuracy_std",
                        "rbf_accuracy_mean",
                        "rbf_accuracy_std",
                        "causal_i_survival_rate_fdr_bh",
                        "causal_i_significant_fdr_bh",
                        "causal_num_edges_evaluated",
                    ]:
                        if key in item and item[key] != "":
                            item[key] = float(item[key])
                    parsed.append(item)
                return parsed

            def read_runs(run_dir: Path) -> dict[int, dict]:
                runs = {}
                for path in sorted((run_dir / "runs").glob("*.json")):
                    run = json.loads(path.read_text())
                    runs[int(run["seed"])] = run
                return runs

            filtered_summary = read_summary(FILTERED_DIR / "summary.csv")
            filtered_n12_summary = read_summary(FILTERED_N12_DIR / "summary.csv")
            no_filter_summary = read_summary(NO_FILTER_DIR / "summary.csv")
            filtered_runs = read_runs(FILTERED_DIR)
            filtered_n12_runs = read_runs(FILTERED_N12_DIR)
            no_filter_runs = read_runs(NO_FILTER_DIR)

            assert sorted(int(row["seed"]) for row in filtered_summary) == sorted(filtered_runs.keys())
            assert sorted(int(row["seed"]) for row in filtered_n12_summary) == sorted(filtered_n12_runs.keys())
            assert sorted(int(row["seed"]) for row in no_filter_summary) == sorted(no_filter_runs.keys())

            seeds = sorted(filtered_runs.keys())
            print("Loaded seeds:", seeds)
            print("Filtered dir:", FILTERED_DIR)
            print("Filtered n=12 dir:", FILTERED_N12_DIR)
            print("No-filter dir:", NO_FILTER_DIR)
            print("Figure output:", FIGURE_DIR)
            """
        ),
        code_cell(
            r"""
            def summarize_metric(rows: list[dict], key: str) -> tuple[float, float]:
                values = [float(row[key]) for row in rows]
                if len(values) == 1:
                    return values[0], 0.0
                return mean(values), stdev(values)

            def filtered_slice_balance(run: dict) -> dict[str, dict[str, float]]:
                split = run["causal_eval"]["split"]["strata_counts"]
                n_per_slice = int(run["num_examples_per_corruption"])
                result = {}
                for slice_name, stats in split.items():
                    retained = int(stats["total"])
                    result[slice_name] = {
                        "input_n": n_per_slice,
                        "retained_n": retained,
                        "retention_rate": retained / n_per_slice if n_per_slice else 0.0,
                    }
                return result

            def no_filter_slice_balance(run: dict) -> dict[str, dict[str, float]]:
                split = run["causal_eval"]["split"]["strata_counts"]
                n_per_slice = int(run["num_examples_per_corruption"])
                result = {}
                for slice_name in split:
                    result[slice_name] = {
                        "input_n": n_per_slice,
                        "retained_n": n_per_slice,
                        "retention_rate": 1.0,
                    }
                return result

            paired = []
            no_filter_by_seed = {int(row["seed"]): row for row in no_filter_summary}
            for row in filtered_summary:
                seed = int(row["seed"])
                paired.append(
                    {
                        "seed": seed,
                        "linear_accuracy": float(row["linear_accuracy_mean"]),
                        "rbf_accuracy": float(row["rbf_accuracy_mean"]),
                        "filtered_survival": float(row["causal_i_survival_rate_fdr_bh"]),
                        "filtered_significant": float(row["causal_i_significant_fdr_bh"]),
                        "no_filter_survival": float(no_filter_by_seed[seed]["causal_i_survival_rate_fdr_bh"]),
                        "no_filter_significant": float(no_filter_by_seed[seed]["causal_i_significant_fdr_bh"]),
                    }
                )

            balance_rows = []
            for seed in seeds:
                filtered_balance = filtered_slice_balance(filtered_runs[seed])
                no_filter_balance = no_filter_slice_balance(no_filter_runs[seed])
                for mode, balance in [("filtered", filtered_balance), ("all_examples", no_filter_balance)]:
                    for slice_name, stats in balance.items():
                        balance_rows.append(
                            {
                                "seed": seed,
                                "mode": mode,
                                "slice": slice_name,
                                "retained_n": stats["retained_n"],
                                "input_n": stats["input_n"],
                                "retention_rate": stats["retention_rate"],
                            }
                        )

            print("Auxiliary classifier summary")
            for metric in ["linear_accuracy_mean", "rbf_accuracy_mean"]:
                m, s = summarize_metric(filtered_summary, metric)
                print(f"  {metric}: mean={m:.4f}, std={s:.4f}")

            print("\\nCausal sensitivity summary")
            for label, rows, key in [
                ("filtered_n12", filtered_n12_summary, "causal_i_survival_rate_fdr_bh"),
                ("filtered", filtered_summary, "causal_i_survival_rate_fdr_bh"),
                ("no_filter", no_filter_summary, "causal_i_survival_rate_fdr_bh"),
            ]:
                m, s = summarize_metric(rows, key)
                print(f"  {label}: I survival mean={m:.4f}, std={s:.4f}")

            def collect_study_points(rows: list[dict], runs: dict[int, dict], label: str, mode: str) -> list[dict]:
                points = []
                for row in rows:
                    seed = int(row["seed"])
                    run = runs[seed]
                    if mode == "filtered":
                        slice_stats = filtered_slice_balance(run)
                        balance_ratio = run["causal_eval"]["split"]["strata_counts"]["ioi:name_swap"]["total"] / max(
                            1, run["causal_eval"]["split"]["strata_counts"]["ioi:abba"]["total"]
                        )
                    else:
                        slice_stats = no_filter_slice_balance(run)
                        balance_ratio = 1.0
                    points.append(
                        {
                            "label": label,
                            "mode": mode,
                            "seed": seed,
                            "num_examples": int(row["num_examples_per_corruption"]),
                            "abba_retention": slice_stats["ioi:abba"]["retention_rate"],
                            "name_swap_retention": slice_stats["ioi:name_swap"]["retention_rate"],
                            "balance_ratio": balance_ratio,
                            "causal_survival": float(row["causal_i_survival_rate_fdr_bh"]),
                            "significant_edges": float(row["causal_i_significant_fdr_bh"]),
                        }
                    )
                return points

            study_points = []
            study_points.extend(collect_study_points(filtered_n12_summary, filtered_n12_runs, label="n=12 filtered", mode="filtered"))
            study_points.extend(collect_study_points(filtered_summary, filtered_runs, label="n=20 filtered", mode="filtered"))
            study_points.extend(collect_study_points(no_filter_summary, no_filter_runs, label="n=20 all examples", mode="all_examples"))
            """
        ),
        markdown_cell(
            """
            ## Figure 1. Auxiliary classical signal

            The classical classifier is not the main claim, but it is useful as a sanity check that the slice graphs contain non-trivial structure.
            """
        ),
        code_cell(
            r"""
            fig, ax = plt.subplots(figsize=(8.2, 4.8))

            x = np.arange(len(seeds))
            width = 0.35
            linear = [row["linear_accuracy"] for row in paired]
            rbf = [row["rbf_accuracy"] for row in paired]

            ax.bar(x - width / 2, linear, width=width, color=COLORS["linear"], label="WL + linear SVM")
            ax.bar(x + width / 2, rbf, width=width, color=COLORS["rbf"], label="WL + RBF SVM")

            ax.axhline(mean(linear), linestyle="--", linewidth=1.6, color=COLORS["linear"], alpha=0.8)
            ax.axhline(mean(rbf), linestyle="--", linewidth=1.6, color=COLORS["rbf"], alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels([f"seed {seed}" for seed in seeds])
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Cross-validated accuracy")
            ax.set_title("Auxiliary classification signal is real but moderate")
            ax.legend(loc="upper right")

            for xpos, value in zip(x - width / 2, linear):
                ax.text(xpos, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
            for xpos, value in zip(x + width / 2, rbf):
                ax.text(xpos, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

            fig.tight_layout()
            fig.savefig(FIGURE_DIR / "figure1_auxiliary_classical_accuracy.png", bbox_inches="tight")
            fig.savefig(FIGURE_DIR / "figure1_auxiliary_classical_accuracy.pdf", bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Figure 2. Causal sensitivity to the `clean_score > base_score` filter

            This is the core paper figure. It shows that the causal signal remains non-trivial without the filter, but the filtered subset is substantially easier and makes the result look much stronger.
            """
        ),
        code_cell(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

            filtered_survival = [row["filtered_survival"] for row in paired]
            no_filter_survival = [row["no_filter_survival"] for row in paired]
            filtered_significant = [row["filtered_significant"] for row in paired]
            no_filter_significant = [row["no_filter_significant"] for row in paired]

            mode_positions = np.array([0.0, 1.0])
            mode_labels = ["All examples", "clean > base"]
            mode_colors = [COLORS["all_examples"], COLORS["filtered"]]
            jitters = np.linspace(-0.06, 0.06, len(seeds))

            panels = [
                (
                    axes[0],
                    no_filter_survival,
                    filtered_survival,
                    "Causal I survival rate (FDR-BH)",
                    "Causal survival",
                    (0.0, 1.08),
                ),
                (
                    axes[1],
                    no_filter_significant,
                    filtered_significant,
                    "Significant causal edges",
                    "Significant edges among top-5 candidates",
                    (0.0, 5.6),
                ),
            ]

            for ax, baseline_values, filtered_values, ylabel, title, ylim in panels:
                means = [mean(baseline_values), mean(filtered_values)]
                ax.bar(
                    mode_positions,
                    means,
                    width=0.55,
                    color=mode_colors,
                    alpha=0.85,
                    edgecolor="none",
                    zorder=1,
                )
                for idx, seed in enumerate(seeds):
                    xs = mode_positions + jitters[idx]
                    ys = [baseline_values[idx], filtered_values[idx]]
                    ax.plot(xs, ys, color="#4a4a4a", linewidth=1.4, alpha=0.6, zorder=2)
                    ax.scatter(xs, ys, s=36, color="white", edgecolor="#2f2f2f", linewidth=1.1, zorder=3)
                for xpos, value in zip(mode_positions, means):
                    ax.text(xpos, value + 0.04 * (ylim[1] - ylim[0]), f"{value:.3f}", ha="center", va="bottom", fontsize=10)
                ax.set_xticks(mode_positions)
                ax.set_xticklabels(mode_labels)
                ax.set_ylim(*ylim)
                ax.set_ylabel(ylabel)
                ax.set_title(title)

            axes[0].legend(
                handles=[
                    plt.Line2D([0], [0], color="#4a4a4a", linewidth=1.4, marker="o", markerfacecolor="white", markeredgecolor="#2f2f2f", label="Per-seed paired change"),
                    plt.Rectangle((0, 0), 1, 1, color=COLORS["all_examples"], alpha=0.85, label="Mean across seeds"),
                ],
                loc="lower right",
            )

            fig.suptitle("The filter materially strengthens the causal result", y=1.03, fontsize=14, fontweight="bold")
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / "figure2_causal_filter_sensitivity.png", bbox_inches="tight")
            fig.savefig(FIGURE_DIR / "figure2_causal_filter_sensitivity.pdf", bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Figure 3. Slice imbalance induced by filtering

            The filtered result must be treated as a secondary analysis because the filter does not preserve slice composition. `name_swap` survives intact; `abba` is heavily depleted.
            """
        ),
        code_cell(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

            x = np.arange(len(seeds))

            filtered_abba = [row["retained_n"] for row in balance_rows if row["mode"] == "filtered" and row["slice"] == "ioi:abba"]
            filtered_name_swap = [row["retained_n"] for row in balance_rows if row["mode"] == "filtered" and row["slice"] == "ioi:name_swap"]

            axes[0].axhline(20, color="#666666", linestyle="--", linewidth=1.5, label="Full slice count (20)")
            axes[0].plot(x, filtered_abba, color=COLORS["abba"], marker="o", linewidth=2.2, markersize=8, label="ABBA")
            axes[0].plot(x, filtered_name_swap, color=COLORS["name_swap"], marker="o", linewidth=2.2, markersize=8, label="Name swap")
            axes[0].fill_between(x, filtered_abba, 20, color=COLORS["abba"], alpha=0.10)

            axes[0].set_xticks(x)
            axes[0].set_xticklabels([f"seed {seed}" for seed in seeds])
            axes[0].set_ylim(0, 22)
            axes[0].set_ylabel("Examples retained after clean > base")
            axes[0].set_title("ABBA is heavily depleted by filtering")
            axes[0].legend(loc="lower right", fontsize=9)

            retention_labels = [SLICE_LABELS["ioi:abba"], SLICE_LABELS["ioi:name_swap"]]
            retention_values = [
                mean(filtered_abba) / 20.0,
                mean(filtered_name_swap) / 20.0,
            ]
            retention_colors = [COLORS["abba"], COLORS["name_swap"]]
            bars = axes[1].bar(retention_labels, retention_values, color=retention_colors)
            axes[1].axhline(1.0, color="#666666", linestyle="--", linewidth=1.4)
            axes[1].set_ylim(0.0, 1.1)
            axes[1].set_ylabel("Mean retention rate under clean > base")
            axes[1].set_title("The filter is strongly slice-dependent")

            for bar, value in zip(bars, retention_values):
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.04,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            fig.tight_layout()
            fig.savefig(FIGURE_DIR / "figure3_filter_slice_balance.png", bbox_inches="tight")
            fig.savefig(FIGURE_DIR / "figure3_filter_slice_balance.pdf", bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Paper-ready takeaway

            The figures support a defensible narrative:

            - The classical graph representation contains non-trivial structure, but classification is not the main result.
            - The causal finding is robust enough to survive the conservative no-filter setting, though at lower strength.
            - The `clean_score > base_score` analysis is informative but not neutral preprocessing; it induces strong slice imbalance and should be reported as a sensitivity/secondary result.
            """
        ),
        markdown_cell(
            """
            ## Figure 4. Why `n=20` is the publication baseline

            This figure compares the two filtered GPT-2 study sizes. The headline causal metric looks equally strong, but the smaller configuration is much less balanced after filtering.
            """
        ),
        code_cell(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

            studies = [
                ("n=12", filtered_n12_summary, filtered_n12_runs),
                ("n=20", filtered_summary, filtered_runs),
            ]
            x = np.arange(len(studies))

            survival_means = [mean(float(row["causal_i_survival_rate_fdr_bh"]) for row in rows) for _, rows, _ in studies]
            imbalance_ratio_means = []
            for _, _, runs in studies:
                ratios = []
                for seed in sorted(runs):
                    strata = runs[seed]["causal_eval"]["split"]["strata_counts"]
                    abba = strata["ioi:abba"]["total"]
                    name_swap = strata["ioi:name_swap"]["total"]
                    ratios.append(name_swap / max(1, abba))
                imbalance_ratio_means.append(mean(ratios))

            axes[0].bar(x, survival_means, width=0.55, color=COLORS["filtered"], alpha=0.85)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([label for label, _, _ in studies])
            axes[0].set_ylim(0.0, 1.08)
            axes[0].set_ylabel("Causal I survival rate (FDR-BH)")
            axes[0].set_title("Headline causal metric")
            for xpos, value in zip(x, survival_means):
                axes[0].text(xpos, value + 0.03, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

            axes[1].bar(x, imbalance_ratio_means, width=0.55, color=COLORS["abba"], alpha=0.9)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([label for label, _, _ in studies])
            axes[1].set_ylim(0.0, 6.0)
            axes[1].set_ylabel("Mean post-selection name_swap / ABBA ratio")
            axes[1].set_title("Smaller n is much more imbalanced")
            for xpos, value in zip(x, imbalance_ratio_means):
                axes[1].text(xpos, value + 0.15, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

            for idx, (_, rows, runs) in enumerate(studies):
                surv_values = [float(row["causal_i_survival_rate_fdr_bh"]) for row in rows]
                ratio_values = []
                for row in rows:
                    strata = runs[int(row["seed"])]["causal_eval"]["split"]["strata_counts"]
                    ratio_values.append(strata["ioi:name_swap"]["total"] / max(1, strata["ioi:abba"]["total"]))
                jitter = np.linspace(-0.06, 0.06, len(surv_values))
                axes[0].scatter(np.full(len(surv_values), idx) + jitter, surv_values, s=34, color="white", edgecolor="#2f2f2f", zorder=3)
                axes[1].scatter(np.full(len(ratio_values), idx) + jitter, ratio_values, s=34, color="white", edgecolor="#2f2f2f", zorder=3)

            fig.suptitle("n=20 is not stronger causally, but it is much more credible", y=1.03, fontsize=14, fontweight="bold")
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / "figure4_n12_vs_n20.png", bbox_inches="tight")
            fig.savefig(FIGURE_DIR / "figure4_n12_vs_n20.pdf", bbox_inches="tight")
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Figure 5. Trade-off between balance and apparent causal strength

            This figure does not claim a smooth law. It shows where each run sits in the space of slice balance and causal strength, making the methodological trade-off visible at a glance.
            """
        ),
        code_cell(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

            style_map = {
                "n=12 filtered": {"color": COLORS["abba"], "marker": "s"},
                "n=20 filtered": {"color": COLORS["filtered"], "marker": "o"},
                "n=20 all examples": {"color": COLORS["all_examples"], "marker": "^"},
            }

            for point in study_points:
                style = style_map[point["label"]]
                axes[0].scatter(
                    point["abba_retention"],
                    point["causal_survival"],
                    s=90,
                    color=style["color"],
                    marker=style["marker"],
                    alpha=0.9,
                )
                axes[1].scatter(
                    point["balance_ratio"],
                    point["significant_edges"],
                    s=90,
                    color=style["color"],
                    marker=style["marker"],
                    alpha=0.9,
                )

            for point in study_points:
                axes[0].annotate(str(point["seed"]), (point["abba_retention"], point["causal_survival"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
                axes[1].annotate(str(point["seed"]), (point["balance_ratio"], point["significant_edges"]), textcoords="offset points", xytext=(5, 5), fontsize=8)

            axes[0].set_xlabel("ABBA retention rate")
            axes[0].set_ylabel("Causal I survival rate (FDR-BH)")
            axes[0].set_title("Retention vs causal survival")
            axes[0].set_xlim(0.0, 1.05)
            axes[0].set_ylim(0.0, 1.08)

            axes[1].set_xlabel("Post-selection name_swap / ABBA ratio")
            axes[1].set_ylabel("Significant causal edges")
            axes[1].set_title("Imbalance ratio vs apparent strength")
            axes[1].set_xlim(0.8, 12.5)
            axes[1].set_ylim(0.0, 5.6)

            legend_handles = [
                plt.Line2D([0], [0], linestyle="", marker=style_map[label]["marker"], color=style_map[label]["color"], markersize=8, label=label)
                for label in ["n=12 filtered", "n=20 filtered", "n=20 all examples"]
            ]
            axes[1].legend(handles=legend_handles, loc="lower right")

            fig.suptitle("Balance and causal strength move together with the analysis choice", y=1.03, fontsize=14, fontweight="bold")
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / "figure5_balance_vs_causal_strength.png", bbox_inches="tight")
            fig.savefig(FIGURE_DIR / "figure5_balance_vs_causal_strength.pdf", bbox_inches="tight")
            plt.show()
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=1))
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
