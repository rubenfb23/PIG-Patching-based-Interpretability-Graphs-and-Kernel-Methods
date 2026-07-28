#!/usr/bin/env python3
"""Build paper tables from experiment results.

Reads run outputs and generates:
- outputs/paper_metrics_for_latex.tex  (LaTeX table rows)
- outputs/paper_metrics_tables.md      (Markdown table)
- outputs/paper_metrics_summary.json   (aggregate summary)

Usage:
    python scripts/build_paper_tables.py --input outputs/classical_publication/gpt2_multiseed_res_k5_n20/results.json
    python scripts/build_paper_tables.py --input outputs/paper_run_v1 --output-dir outputs
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    v = [x for x in values if x is not None]
    if not v:
        return float("nan"), float("nan")
    m = float(sum(v) / len(v))
    s = float((sum((x - m) ** 2 for x in v) / len(v)) ** 0.5)
    return m, s


def _bootstrap_ci(values: list[float], alpha: float = 0.05, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        return float(values[0]) if values else (float("nan"), float("nan"), float("nan"))
    m = float(sum(values) / len(values))
    rng = __import__("numpy").random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)))
    boot_means = draws.mean(axis=1)
    low = float(__import__("numpy").quantile(boot_means, alpha / 2))
    high = float(__import__("numpy").quantile(boot_means, 1 - alpha / 2))
    return m, low, high


def _parse_run_key(key: str) -> dict:
    """Parse run_key into components."""
    parts = key.replace("_", " ").replace("seed", "SEED").split()
    result = {}
    for p in parts:
        if p.startswith("SEED"):
            result["seed"] = int(p[4:])
        elif p.startswith("k"):
            result["k"] = int(p[1:])
        elif p.startswith("n"):
            result["n"] = int(p[1:])
    return result


def build_tables_from_results(results_path: str) -> dict:
    """Build tables from a single results.json file."""
    with open(results_path) as f:
        data = json.load(f)

    runs = data.get("runs", [])
    tables = {
        "main": [],  # (representation, acc, std, dim, notes)
        "scalable": [],  # (representation, dim, n100_acc, n100_std, n500_acc, n500_std)
        "null_controls": [],  # (representation, control, acc, std)
        "causal": [],  # (edge_id, I_mean, M_mean, classification)
    }

    by_n: dict[int, list] = {}
    for run in runs:
        n = run.get("n", 0)
        by_n.setdefault(n, []).append(run)

    # Collect per-n aggregates
    for n in sorted(by_n.keys()):
        seeds_acc = {}  # feature -> list of acc values
        for run in by_n[n]:
            seed = run.get("seed")

            # Fixed layout
            fl = run.get("fixed_layout", {})
            fl_obs = fl.get("observed", {})
            fl_lin = fl_obs.get("linear", {})
            fl_rbf = fl_obs.get("rbf", {})
            if fl_lin.get("accuracy_mean") is not None:
                seeds_acc.setdefault("fixed_layout_signed", []).append(fl_lin["accuracy_mean"])

            # WL bootstrap
            wl = run.get("wl_bootstrap", {})
            wl_obs = wl.get("observed", {})
            wl_lin = wl_obs.get("linear", {})
            if wl_lin.get("accuracy_mean") is not None:
                seeds_acc.setdefault("wl_bootstrap", []).append(wl_lin["accuracy_mean"])

            # WL per-example
            wle = run.get("wl_per_example", {})
            wle_obs = wle.get("observed", {})
            wle_lin = wle_obs.get("linear", {})
            if wle_lin.get("accuracy_mean") is not None:
                seeds_acc.setdefault("wl_per_example", []).append(wle_lin["accuracy_mean"])

            # Patch-effect node vector
            pe = run.get("patch_effect", {})
            pe_obs = pe.get("observed", {})
            pe_lin = pe_obs.get("linear", {})
            if pe_lin.get("accuracy_mean") is not None:
                seeds_acc.setdefault("patch_effect_node_vector", []).append(pe_lin["accuracy_mean"])

            # Top-k node identity
            tk = run.get("topk_node", {})
            tk_obs = tk.get("observed", {})
            tk_lin = tk_obs.get("linear", {})
            if tk_lin.get("accuracy_mean") is not None:
                seeds_acc.setdefault("topk_node", []).append(tk_lin["accuracy_mean"])

            # Global histogram
            hg = run.get("histogram", {})
            hg_obs = hg.get("observed", {})
            hg_lin = hg_obs.get("linear", {})
            if hg_lin.get("accuracy_mean") is not None:
                seeds_acc.setdefault("global_histogram", []).append(hg_lin["accuracy_mean"])

            # Hashed fixed-layout
            hd = run.get("hash", {})
            hd_obs = hd.get("observed", {})
            hd_lin = hd_obs.get("linear", {})
            if hd_lin.get("accuracy_mean") is not None:
                seeds_acc.setdefault("hashed_fixed_layout", []).append(hd_lin["accuracy_mean"])

            # Coarse count
            cc = run.get("coarse_count", {})
            cc_obs = cc.get("observed", {})
            cc_lin = cc_obs.get("linear", {})
            if cc_lin.get("accuracy_mean") is not None:
                seeds_acc.setdefault("coarse_count", []).append(cc_lin["accuracy_mean"])

            # Label permutation
            lp = run.get("label_permutation", {})
            lp_obs = lp.get("observed", {})
            lp_lin = lp_obs.get("linear", {})
            if lp_lin.get("accuracy_mean") is not None:
                seeds_acc.setdefault("label_permutation", []).append(lp_lin["accuracy_mean"])

            # Random graph
            rg = run.get("random_graph", {})
            rg_obs = rg.get("observed", {})
            rg_lin = rg_obs.get("linear", {})
            if rg_lin.get("accuracy_mean") is not None:
                seeds_acc.setdefault("random_graph_matched_degree", []).append(rg_lin["accuracy_mean"])

            # Null controls
            for feat_name in ["fixed_layout", "wl_bootstrap"]:
                bl = run.get(feat_name, {})
                nc = bl.get("null_controls", {})
                for ctrl_name, ctrl_data in nc.items():
                    for kernel in ["linear", "rbf"]:
                        cd = ctrl_data.get(kernel, {})
                        acc = cd.get("accuracy_mean")
                        if acc is not None:
                            tables["null_controls"].append({
                                "feature": feat_name,
                                "control": ctrl_name,
                                "kernel": kernel,
                                "n": n,
                                "seed": seed,
                                "accuracy_mean": acc,
                                "accuracy_std": cd.get("accuracy_std", 0),
                            })

            # Causal
            ce = run.get("causal_eval", {})
            for edge in ce.get("edge_summaries", []):
                tables["causal"].append({
                    "n": n,
                    "seed": seed,
                    "edge_id": edge.get("edge_id", ""),
                    "I_mean": edge.get("I_mean"),
                    "M_mean": edge.get("M_mean"),
                    "classification": edge.get("classification", ""),
                })

    # Build main table
    dim_map = {
        "fixed_layout_signed": "N(N-1)",
        "wl_bootstrap": "WL-H3",
        "wl_per_example": "WL-H3",
        "patch_effect_node_vector": "N×D",
        "topk_node": "k=10",
        "global_histogram": "2N+6",
        "hashed_fixed_layout": "d=1024",
        "coarse_count": "≈105",
        "label_permutation": "N(N-1)",
        "random_graph_matched_degree": "N(N-1)",
    }
    for feat, accs in sorted(seeds_acc.items()):
        m, s = _mean_std(accs)
        ci_low, ci_high = float("nan"), float("nan")
        if len(accs) >= 2:
            _, ci_low, ci_high = _bootstrap_ci(accs)
        tables["main"].append({
            "representation": _feat_label(feat),
            "accuracy_mean": m,
            "accuracy_std": s,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "dim": dim_map.get(feat, "?"),
            "n": sorted(by_n.keys())[0] if by_n else 0,
            "n_seeds": len(accs),
        })

    # Compute scalable table (need n=100 and n=500)
    n100 = seeds_acc.get("fixed_layout_signed", [])
    n100_h = []
    n500 = []
    n500_h = []
    # Will be populated when we have multiple n values
    for feat in ["fixed_layout_signed", "wl_bootstrap"]:
        if feat in seeds_acc:
            pass  # handled below

    # Aggregate by n
    for n in sorted(by_n.keys()):
        feat_accs: dict[str, list] = {}
        for run in by_n[n]:
            fl = run.get("fixed_layout", {})
            fl_obs = fl.get("observed", {})
            fl_lin = fl_obs.get("linear", {})
            if fl_lin.get("accuracy_mean") is not None:
                feat_accs.setdefault("fixed_layout", []).append(fl_lin["accuracy_mean"])
            wl = run.get("wl_bootstrap", {})
            wl_obs = wl.get("observed", {})
            wl_lin = wl_obs.get("linear", {})
            if wl_lin.get("accuracy_mean") is not None:
                feat_accs.setdefault("wl_bootstrap", []).append(wl_lin["accuracy_mean"])
        for feat, accs in feat_accs.items():
            m, s = _mean_std(accs)
            if n == 100:
                tables["scalable"].append({"representation": _feat_label(feat), "dim": "varies", "n": n, "acc": m, "std": s})
            elif n == 500:
                tables["scalable"].append({"representation": _feat_label(feat), "dim": "varies", "n": n, "acc": m, "std": s})

    return tables


def _feat_label(name: str) -> str:
    labels = {
        "fixed_layout_signed": "Fixed layout signed",
        "wl_bootstrap": "WL bootstrap",
        "wl_per_example": "WL per-example",
        "hashed_fixed_layout": "Hashed fixed-layout",
        "coarse_count": "Coarse count",
        "patch_effect_node_vector": "Patch-effect node vector",
        "topk_node": "Top-k node identity",
        "global_histogram": "Global histogram",
        "label_permutation": "Label permutation",
        "random_graph_matched_degree": "Random graph (matched deg.)",
    }
    return labels.get(name, name)


def tex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")


def tables_to_latex(tables: dict) -> str:
    """Convert tables dict to LaTeX rows."""
    lines = []

    # Main table
    lines.append("% Main results rows")
    for row in tables["main"]:
        acc_str = f"${row['accuracy_mean']:.4f} \\pm {row['accuracy_std']:.4f}$"
        ci_str = f"CI$_{{95}}$=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]" if not math.isnan(row['ci_low']) else ""
        lines.append(
            f"{tex_escape(row['representation'])} ($n\!={row['n']}$) & {acc_str} & {row['dim']} & {row['n_seeds']} seeds & {ci_str} \\\\"
        )

    lines.append("")
    lines.append("% Scalable representations")
    for row in tables["scalable"]:
        lines.append(
            f"{tex_escape(row['representation'])} ($n\!={row['n']}$) & {row['dim']} & {row['acc']:.4f} & {row['std']:.4f} \\\\"
        )

    lines.append("")
    lines.append("% Null controls")
    null_groups: dict[tuple[str, str], list] = {}
    for row in tables["null_controls"]:
        key = (row["feature"], row["control"])
        null_groups.setdefault(key, []).append(row)
    for (feat, ctrl), rows in sorted(null_groups.items()):
        accs = [r["accuracy_mean"] for r in rows if r["accuracy_mean"] is not None]
        m, s = _mean_std(accs)
        lines.append(
            f"{tex_escape(ctrl)} ({tex_escape(feat)}) & {m:.4f} & {s:.4f} & {len(accs)} repeats \\\\"
        )

    lines.append("")
    lines.append("% Causal validation")
    causal_groups: dict[str, list] = {}
    for row in tables["causal"]:
        causal_groups.setdefault(row["classification"], []).append(row)
    for cls, rows in sorted(causal_groups.items()):
        I_vals = [r["I_mean"] for r in rows if r["I_mean"] is not None]
        M_vals = [r["M_mean"] for r in rows if r["M_mean"] is not None]
        I_m, I_s = _mean_std(I_vals)
        M_m, M_s = _mean_std(M_vals)
        lines.append(
            f"{tex_escape(cls)} & {len(rows)} edges & I$_{{mean}}$={I_m:.3f} & M$_{{mean}}$={M_m:.3f} \\\\"
        )

    return "\n".join(lines)


def tables_to_markdown(tables: dict) -> dict:
    """Convert tables dict to structured markdown data."""
    result = {
        "main_table": tables["main"],
        "scalable_table": tables["scalable"],
        "null_controls": tables["null_controls"],
        "causal_table": tables["causal"],
    }

    # Artifact audit
    result["artifact_audit"] = {
        "main_results": {
            "exists": len(tables["main"]) > 0,
            "n_values": list(set(r["n"] for r in tables["main"])),
            "seeds_per_n": {r["n"]: r["n_seeds"] for r in tables["main"]},
        },
        "causal": {
            "exists": len(tables["causal"]) > 0,
            "edges_total": len(tables["causal"]),
        },
        "null_controls": {
            "exists": len(tables["null_controls"]) > 0,
            "controls": list(set((r["feature"], r["control"]) for r in tables["null_controls"])),
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Build paper tables from results")
    parser.add_argument("--input", "-i", nargs="*", help="Input results.json file(s) or directory")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Output directory")
    args = parser.parse_args()

    input_paths = args.input or ["outputs/classical_publication"]
    all_tables = {"main": [], "scalable": [], "null_controls": [], "causal": []}

    for p in input_paths:
        path = Path(p)
        if path.is_dir():
            for json_file in sorted(path.glob("**/results.json")):
                tables = build_tables_from_results(str(json_file))
                for key in all_tables:
                    all_tables[key].extend(tables[key])
        elif path.is_file():
            tables = build_tables_from_results(str(path))
            for key in all_tables:
                all_tables[key].extend(tables[key])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write LaTeX
    latex = tables_to_latex(all_tables)
    (output_dir / "paper_metrics_for_latex.tex").write_text(latex)

    # Write markdown
    md = tables_to_markdown(all_tables)
    (output_dir / "paper_metrics_summary.json").write_text(json.dumps(md, indent=2))

    # Write markdown table
    md_lines = ["# Paper Metrics Tables\n\n"]
    md_lines.append("## Main Table\n\n")
    md_lines.append("| Representation | Acc | Std | Dim | Seeds | CI$_{95}$ |\n")
    md_lines.append("| --- | --- | --- | --- | --- | --- |\n")
    for r in all_tables["main"]:
        ci = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]" if not math.isnan(r['ci_low']) else "N/A"
        md_lines.append(
            f"| {tex_escape(r['representation'])} ($n={r['n']}$) | {r['accuracy_mean']:.4f} | {r['accuracy_std']:.4f} | {r['dim']} | {r['n_seeds']} | {ci} |\n"
        )

    (output_dir / "paper_metrics_tables_toy.md").write_text("".join(md_lines))

    print(f"Written to {output_dir}:")
    print(f"  - paper_metrics_for_latex.tex ({len(all_tables['main'])} main rows)")
    print(f"  - paper_metrics_summary.json (structured)")
    print(f"  - paper_metrics_tables_toy.md (markdown)")


if __name__ == "__main__":
    main()

# ---- ---- ---- ---- ---- ---- ----
# NeurIPS revision mode extensions
# ---- ---- ---- ---- ---- ---- ----

def compute_statistical_tests(values_a: list[float], values_b: list[float], seed: int = 42) -> dict:
    """Compute Wilcoxon signed-rank test and bootstrap CI between two sets of values.

    Args:
        values_a: Accuracy values for method A (e.g., fixed_layout_signed).
        values_b: Accuracy values for method B (e.g., PCA projected).
        seed: Random seed for bootstrap.

    Returns:
        Dict with test statistics.
    """
    import scipy.stats as stats

    result = {}
    n = min(len(values_a), len(values_b))
    if n < 2:
        result["wilcoxon"] = {"p_value": float("nan"), "z_score": float("nan")}
        result["bootstrap_ci"] = {"diff_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        return result

    a = values_a[:n]
    b = values_b[:n]
    diff = [ai - bi for ai, bi in zip(a, b)]

    # Wilcoxon signed-rank test
    wilcoxon_result = stats.wilcoxon(a, b)
    result["wilcoxon"] = {
        "p_value": float(wilcoxon_result.pvalue),
        "z_score": float(wilcoxon_result.statistic) if hasattr(wilcoxon_result, "statistic") else None,
        "n_pairs": n,
    }

    # Mean difference
    mean_diff = float(sum(diff) / len(diff))
    result["bootstrap_ci"] = {
        "diff_mean": mean_diff,
    }

    # Bootstrap CI 95%
    n_boot = 5000
    rng = __import__("numpy").random.default_rng(seed)
    boot_diffs = rng.choice(diff, size=(n_boot, n)).mean(axis=1)
    ci_low = float(__import__("numpy").quantile(boot_diffs, 0.025))
    ci_high = float(__import__("numpy").quantile(boot_diffs, 0.975))
    result["bootstrap_ci"]["ci_low"] = ci_low
    result["bootstrap_ci"]["ci_high"] = ci_high
    result["bootstrap_ci"]["n_bootstraps"] = n_boot

    # Also compute mean/std for each method
    result["method_a"] = {"mean": float(sum(a) / len(a)), "std": float(__import__("numpy").std(a)), "n": len(a)}
    result["method_b"] = {"mean": float(sum(b) / len(b)), "std": float(__import__("numpy").std(b)), "n": len(b)}

    return result


def compute_wilcoxon_vs_chance_table(seed_acc_dict: dict[str, list[float]], chance: float = 0.5, seed: int = 42) -> dict:
    """Compute Wilcoxon signed-rank test against chance level for each representation.

    Args:
        seed_acc_dict: Dict mapping representation name -> list of seed accuracies.
        chance: Chance level (default 0.5 for binary classification).
        seed: Random seed for bootstrap.

    Returns:
        Dict with test results per representation.
    """
    import scipy.stats as stats

    results = {}
    for name, accs in seed_acc_dict.items():
        n = len(accs)
        if n < 2:
            results[name] = {"n": n, "wilcoxon_p": float("nan"), "z_score": float("nan")}
            continue

        # Test against chance level
        wilcoxon_result = stats.wilcoxon([a - chance for a in accs])
        mean_acc = float(sum(accs) / n)
        std_acc = float(__import__("numpy").std(accs))

        # Bootstrap CI for mean accuracy
        rng = __import__("numpy").random.default_rng(seed)
        boot_means = rng.choice(accs, size=(5000, n)).mean(axis=1)
        ci_low = float(__import__("numpy").quantile(boot_means, 0.025))
        ci_high = float(__import__("numpy").quantile(boot_means, 0.975))

        results[name] = {
            "n": n,
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "wilcoxon_p": float(wilcoxon_result.pvalue),
            "z_score": float(wilcoxon_result.statistic) if hasattr(wilcoxon_result, "statistic") else None,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
        }
    return results


def build_n_curve_table(n_grid_results: dict) -> dict:
    """Build accuracy vs n data from n_grid_results.

    Args:
        n_grid_results: Dict mapping n -> {feature_name -> result} for a single seed.

    Returns:
        Dict mapping (task, seed) -> {n -> {feature -> acc}}.
    """
    curve = {}
    for feat_name, feat_data in n_grid_results.items():
        if feat_name == "task" or feat_name == "seed":
            continue
        obs = feat_data.get("observed", {})
        for kernel in ("linear", "rbf"):
            acc = obs.get(kernel, {}).get("accuracy_mean")
            if acc is not None:
                curve[(feat_name, kernel)] = acc
    return curve


def build_all_n_curves(reports: list[dict]) -> dict:
    """Extract accuracy vs n curves across all tasks, seeds, and features."""
    # Structure: {task -> feature -> kernel -> seed -> n -> acc}
    curves: dict[str, dict[str, dict[str, dict[int, dict[int, float]]]]] = {}

    for report in reports:
        task = report.get("task", "unknown")
        seed = report.get("seed", 0)
        n_results = report.get("n_grid_results", {})

        for n, feat_dict in n_results.items():
            for feat_name, feat_data in feat_dict.items():
                if feat_name in ("task", "seed", "ablation_results"):
                    continue
                obs = feat_data.get("observed", {})
                for kernel in ("linear", "rbf"):
                    acc = obs.get(kernel, {}).get("accuracy_mean")
                    if acc is None:
                        continue

                    curves.setdefault(task, {}).setdefault(feat_name, {}).setdefault(kernel, {}).setdefault(seed, {})[n] = acc

    return curves


def aggregate_n_curves(curves: dict) -> dict:
    """Aggregate n-curve data: for each (task, feature, kernel), compute mean/std across seeds at each n."""
    aggregated: dict[str, dict[str, dict[str, dict[int, tuple[float, float]]]]] = {}

    for task, feat_dict in curves.items():
        for feat_name, kernel_dict in feat_dict.items():
            for kernel, seed_dict in kernel_dict.items():
                # Group by n
                n_to_accs: dict[int, list[float]] = {}
                for seed, n_dict in seed_dict.items():
                    for n, acc in n_dict.items():
                        n_to_accs.setdefault(n, []).append(acc)

                for n, accs in n_to_accs.items():
                    if not accs:
                        continue
                    mean = float(sum(accs) / len(accs))
                    std = float(__import__("numpy").std(accs)) if len(accs) > 1 else 0.0
                    aggregated.setdefault(task, {}).setdefault(feat_name, {}).setdefault(kernel, {})[n] = (mean, std)

    return aggregated


def build_k_ablation_table(reports: list[dict]) -> dict:
    """Extract k ablation results."""
    results = {}
    for report in reports:
        task = report.get("task", "unknown")
        seed = report.get("seed", 0)
        ablations = report.get("ablation_results", {})
        k_data = ablations.get("k_ablation", {})
        for feat_name, data in k_data.items():
            acc = data.get("acc_linear")
            if acc is not None:
                key = f"{task}_seed{seed}_{feat_name}"
                results[key] = {"k": data["k"], "acc_linear": acc, "acc_rbf": data.get("acc_rbf"), "seed": seed, "task": task}
    return results


def build_wl_depth_table(reports: list[dict]) -> dict:
    """Extract WL depth ablation results."""
    results = {}
    for report in reports:
        task = report.get("task", "unknown")
        seed = report.get("seed", 0)
        ablations = report.get("ablation_results", {})
        wl_data = ablations.get("wl_depth_ablation", {})
        for feat_name, data in wl_data.items():
            acc = data.get("acc_linear")
            if acc is not None:
                key = f"{task}_seed{seed}_{feat_name}"
                results[key] = {"depth": data["depth"], "acc_linear": acc, "acc_rbf": data.get("acc_rbf"), "seed": seed, "task": task}
    return results


def build_direction_ablation_table(reports: list[dict]) -> dict:
    """Extract direction constraint ablation results."""
    results = {}
    for report in reports:
        task = report.get("task", "unknown")
        seed = report.get("seed", 0)
        ablations = report.get("ablation_results", {})
        dir_data = ablations.get("direction_ablation", {})
        for feat_name, data in dir_data.items():
            acc = data.get("acc_linear")
            if acc is not None:
                key = f"{task}_seed{seed}_{feat_name}"
                results[key] = {"enforce_direction": data["enforce_direction"], "acc_linear": acc, "acc_rbf": data.get("acc_rbf"), "seed": seed, "task": task}
    return results


def build_dim_control_table(reports: list[dict]) -> dict:
    """Extract dimensionality control results (PCA vs RP vs spectral)."""
    results = {}
    for report in reports:
        task = report.get("task", "unknown")
        seed = report.get("seed", 0)
        n_results = report.get("n_grid_results", {})
        for n, feat_dict in n_results.items():
            for feat_name in ("fixed_layout_signed_pca96", "fixed_layout_signed_rp96", "fixed_layout_signed"):
                if feat_name not in feat_dict:
                    continue
                data = feat_dict[feat_name]
                obs = data.get("observed", {})
                acc_lin = obs.get("linear", {}).get("accuracy_mean")
                acc_rbf = obs.get("rbf", {}).get("accuracy_mean")
                if acc_lin is not None:
                    key = f"{task}_seed{seed}_{n}_{feat_name}"
                    results[key] = {
                        "n": n,
                        "feature": feat_name,
                        "acc_linear": acc_lin,
                        "acc_rbf": acc_rbf,
                        "seed": seed,
                        "task": task,
                    }
    return results


def build_n_curve_latex(aggregated: dict, task: str = "ioi") -> str:
    """Build LaTeX table for accuracy vs n learning curve."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Accuracy vs number of prompt pairs ($n$).}")
    lines.append("\\begin{tabular}{l" + "ccc" * 3 + "}")
    lines.append("\\hline")
    lines.append("\\textbf{n} & \\multicolumn{2}{c}{\\textbf{Fixed layout}} & \\multicolumn{2}{c}{\\textbf{WL bootstrap}} & \\multicolumn{2}{c}{\\textbf{Patch-effect}} \\\\")
    lines.append("\\textbf{} & \\textbf{acc} & \\textbf{std} & \\textbf{acc} & \\textbf{std} & \\textbf{acc} & \\textbf{std} \\\\ \\hline")

    # Find common n values
    feat_names = list(aggregated.get(task, {}).keys())
    if not feat_names:
        return lines

    all_ns = set()
    for feat in feat_names:
        for n in aggregated[task].get(feat, {}).get("linear", {}).keys():
            all_ns.add(n)

    for n in sorted(all_ns):
        row_parts = [f"\\textbf{{{n}}}]"]
        for feat in ["fixed_layout_signed", "wl_bootstrap", "patch_effect_node_vector"]:
            lin_data = aggregated.get(task, {}).get(feat, {}).get("linear", {}).get(n)
            if lin_data:
                mean, std = lin_data
                row_parts.append(f"{mean:.4f} & {std:.4f}")
            else:
                row_parts.append("-- & --")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_k_ablation_latex(k_results: dict) -> str:
    """Build LaTeX table for k ablation."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{k ablation results (accuracy vs k).}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\hline")
    lines.append("\\textbf{k} & \\textbf{acc (linear)} & \\textbf{acc (rbf)} \\\\ \\hline")
    for key, data in sorted(k_results.items(), key=lambda x: x[1]["k"]):
        lines.append(f"{data['k']} & {data['acc_linear']:.4f} & {data.get('acc_rbf') or 0:.4f} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_wl_depth_latex(wl_results: dict) -> str:
    """Build LaTeX table for WL depth ablation."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{WL depth ablation results (accuracy vs WL depth).}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\hline")
    lines.append("\\textbf{depth} & \\textbf{acc (linear)} & \\textbf{acc (rbf)} \\\\ \\hline")
    for key, data in sorted(wl_results.items(), key=lambda x: x[1]["depth"]):
        lines.append(f"{data['depth']} & {data['acc_linear']:.4f} & {data.get('acc_rbf') or 0:.4f} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_direction_ablation_latex(dir_results: dict) -> str:
    """Build LaTeX table for direction constraint ablation."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Direction constraint ablation results.}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\hline")
    lines.append("\\textbf{direction} & \\textbf{acc (linear)} & \\textbf{acc (rbf)} \\\\ \\hline")
    for key, data in sorted(dir_results.items(), key=lambda x: x[1]["enforce_direction"], reverse=True):
        lines.append(f"{'enforced' if data['enforce_direction'] else 'free'} & {data['acc_linear']:.4f} & {data.get('acc_rbf') or 0:.4f} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_dim_control_latex(dim_results: dict) -> str:
    """Build LaTeX table for dimensionality control (PCA vs RP)."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Dimensionality control: PCA vs Random Projection (96 dims).}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\hline")
    lines.append("\\textbf{feature} & \\textbf{acc (linear)} & \\textbf{acc (rbf)} & \\textbf{dim} \\\\ \\hline")

    # Group by feature
    feat_groups: dict[str, list] = {}
    for key, data in dim_results.items():
        feat_groups.setdefault(data["feature"], []).append(data)

    for feat in ["fixed_layout_signed", "fixed_layout_signed_pca96", "fixed_layout_signed_rp96"]:
        if feat not in feat_groups:
            continue
        accs_lin = [d["acc_linear"] for d in feat_groups[feat] if d["acc_linear"] is not None]
        accs_rbf = [d["acc_rbf"] for d in feat_groups[feat] if d["acc_rbf"] is not None]
        mean_lin = float(sum(accs_lin) / len(accs_lin)) if accs_lin else 0
        std_lin = float(__import__("numpy").std(accs_lin)) if len(accs_lin) > 1 else 0
        mean_rbf = float(sum(accs_rbf) / len(accs_rbf)) if accs_rbf else 0
        std_rbf = float(__import__("numpy").std(accs_rbf)) if len(accs_rbf) > 1 else 0

        dim_label = "14,280" if feat == "fixed_layout_signed" else "96"
        lines.append(f"{feat} & {mean_lin:.4f} & {mean_rbf:.4f} & {dim_label} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_statistical_tests_latex(test_results: dict) -> str:
    """Build LaTeX table for statistical tests."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Statistical tests: Wilcoxon signed-rank and bootstrap CI (95\\%).}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\hline")
    lines.append("\\textbf{comparison} & \\textbf{$\\Delta$} & \\textbf{CI$_{low}$} & \\textbf{CI$_{high}$} & \\textbf{p-value} & \\textbf{n} \\\\ \\hline")

    for name, test in test_results.items():
        wilcoxon = test.get("wilcoxon", {})
        ci = test.get("bootstrap_ci", {})
        p_val = wilcoxon.get("p_value", float("nan"))
        delta = ci.get("diff_mean", float("nan"))
        ci_low = ci.get("ci_low", float("nan"))
        ci_high = ci.get("ci_high", float("nan"))
        n_pairs = wilcoxon.get("n_pairs", 0)

        p_str = f"{p_val:.4f}" if not (isinstance(p_val, float) and (p_val != p_val)) else "--"
        delta_str = f"{delta:.4f}" if not (isinstance(delta, float) and (delta != delta)) else "--"
        ci_low_str = f"{ci_low:.4f}" if not (isinstance(ci_low, float) and (ci_low != ci_low)) else "--"
        ci_high_str = f"{ci_high:.4f}" if not (isinstance(ci_high, float) and (ci_high != ci_high)) else "--"

        pair_name = name.replace("_vs_", " vs ")
        lines.append(f"{pair_name} & {delta_str} & {ci_low_str} & {ci_high_str} & {p_str} & {n_pairs} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def load_revision_results(input_dir: str) -> dict:
    """Load revision results from runs/ directory.

    Args:
        input_dir: Directory containing neurips_revision runs.

    Returns:
        Dict with tasks, seeds, and aggregated data.
    """
    input_path = Path(input_dir)
    reports = []

    # Load all JSON files from runs/
    runs_dir = input_path / "runs"
    if runs_dir.exists():
        for json_file in sorted(runs_dir.glob("**/seed*.json")):
            with json_file.open() as f:
                reports.append(json.load(f))

    # Also try loading from results.json
    results_file = input_path / "results.json"
    if results_file.exists():
        with results_file.open() as f:
            data = json.load(f)
            if "runs" in data:
                reports = data["runs"]

    return {"reports": reports, "input_dir": str(input_path)}


def main():
    parser = argparse.ArgumentParser(description="Build paper tables from results")
    parser.add_argument("--input", "-i", nargs="*", help="Input results.json file(s) or directory")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Output directory")
    parser.add_argument("--revision-mode", action="store_true", help="Enable NeurIPS revision mode with n-curve and ablation tables")
    parser.add_argument("--task", default=None, help="Specific task to build tables for")
    args = parser.parse_args()

    input_paths = args.input or ["outputs/classical_publication"]
    all_tables = {"main": [], "scalable": [], "null_controls": [], "causal": []}

    for p in input_paths:
        path = Path(p)
        if path.is_dir():
            for json_file in sorted(path.glob("**/results.json")):
                tables = build_tables_from_results(str(json_file))
                for key in all_tables:
                    all_tables[key].extend(tables[key])
        elif path.is_file():
            tables = build_tables_from_results(str(path))
            for key in all_tables:
                all_tables[key].extend(tables[key])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write standard LaTeX
    latex = tables_to_latex(all_tables)
    (output_dir / "paper_metrics_for_latex.tex").write_text(latex)

    # Write standard markdown
    md = tables_to_markdown(all_tables)
    (output_dir / "paper_metrics_summary.json").write_text(json.dumps(md, indent=2))

    # Write standard markdown table
    md_lines = ["# Paper Metrics Tables\n\n"]
    md_lines.append("## Main Table\n\n")
    md_lines.append("| Representation | Acc | Std | Dim | Seeds | CI$_{95}$ |\n")
    md_lines.append("| --- | --- | --- | --- | --- | --- |\n")
    for r in all_tables["main"]:
        ci = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]" if not math.isnan(r['ci_low']) else "N/A"
        md_lines.append(
            f"| {tex_escape(r['representation'])} ($n={r['n']}$) | {r['accuracy_mean']:.4f} | {r['accuracy_std']:.4f} | {r['dim']} | {r['n_seeds']} | {ci} |\n"
        )

    (output_dir / "paper_metrics_tables_toy.md").write_text("".join(md_lines))

    print(f"Written to {output_dir}:")
    print(f"  - paper_metrics_for_latex.tex ({len(all_tables['main'])} main rows)")
    print(f"  - paper_metrics_summary.json (structured)")
    print(f"  - paper_metrics_tables_toy.md (markdown)")

    # ---- Revision mode ----
    if args.revision_mode:
        print("\n=== Revision Mode ===")

        # Load revision results
        rev_data = load_revision_results(str(Path(input_paths[0]) if input_paths else "outputs/neurips_revision"))
        reports = rev_data["reports"]
        input_dir = rev_data["input_dir"]

        if not reports:
            print("No revision reports found.")
        else:
            # Filter by task if specified
            if args.task:
                reports = [r for r in reports if r.get("task") == args.task]
                print(f"Filtered to task: {args.task} ({len(reports)} reports)")

            print(f"Total revision reports: {len(reports)}")

            # ---- n-curve data ----
            print("\nBuilding n-curve data...")
            curves = build_all_n_curves(reports)
            aggregated = aggregate_n_curves(curves)

            # Save n-curve JSON
            n_curve_data = {}
            for task, feat_dict in aggregated.items():
                for feat_name, kernel_dict in feat_dict.items():
                    for kernel, n_dict in kernel_dict.items():
                        n_curve_data.setdefault(task, {}).setdefault(feat_name, {}).setdefault(kernel, {})
                        for n, (mean, std) in n_dict.items():
                            n_curve_data[task][feat_name][kernel][n] = {"mean": mean, "std": std}

            n_curve_path = output_dir / "n_curve_data.json"
            n_curve_path.write_text(json.dumps(n_curve_data, indent=2))
            print(f"  Saved n-curve data to {n_curve_path}")

            # ---- Statistical tests ----
            print("Computing statistical tests...")
            stat_tests = {}

            # Compare fixed_layout_signed vs chance (0.5) for each task
            for task, feat_dict in curves.items():
                fixed_layout_data = feat_dict.get("fixed_layout_signed", {})
                if "linear" in fixed_layout_data:
                    # Collect all seed accuracies
                    all_accs = []
                    for seed_accs in fixed_layout_data["linear"].values():
                        all_accs.extend(seed_accs.values())
                    if all_accs:
                        test = compute_wilcoxon_vs_chance_table({f"{task}_fixed_layout": all_accs}, chance=0.5)
                        stat_tests[f"{task}_fixed_layout_vs_chance"] = test[f"{task}_fixed_layout"]

            # Compare PCA vs RP for each task
            for task, feat_dict in curves.items():
                pca_data = feat_dict.get("fixed_layout_signed_pca96", {})
                rp_data = feat_dict.get("fixed_layout_signed_rp96", {})
                if "linear" in pca_data and "linear" in rp_data:
                    # Match by seed
                    common_seeds = set(pca_data["linear"].keys()) & set(rp_data["linear"].keys())
                    pca_accs = []
                    rp_accs = []
                    for seed in sorted(common_seeds):
                        pca_accs.append(pca_data["linear"][seed])
                        rp_accs.append(rp_data["linear"][seed])
                    if pca_accs and rp_accs:
                        test = compute_statistical_tests(pca_accs, rp_accs)
                        stat_tests[f"{task}_pca96_vs_rp96"] = test

            stat_tests_path = output_dir / "statistical_tests.json"
            stat_tests_path.write_text(json.dumps(stat_tests, indent=2))
            print(f"  Saved statistical tests to {stat_tests_path}")

            # ---- Ablation tables ----
            print("Building ablation tables...")
            k_results = build_k_ablation_table(reports)
            wl_results = build_wl_depth_table(reports)
            dir_results = build_direction_ablation_table(reports)
            dim_results = build_dim_control_table(reports)

            # Save ablation JSONs
            (output_dir / "k_ablation_data.json").write_text(json.dumps(k_results, indent=2))
            (output_dir / "wl_depth_data.json").write_text(json.dumps(wl_results, indent=2))
            (output_dir / "direction_ablation_data.json").write_text(json.dumps(dir_results, indent=2))
            (output_dir / "dim_control_data.json").write_text(json.dumps(dim_results, indent=2))

            # ---- Write LaTeX tables ----
            # N-curve LaTeX
            for task in aggregated:
                latex = build_n_curve_latex(aggregated, task)
                (output_dir / f"n_curve_latex_{task}.tex").write_text(latex)
                print(f"  Saved n-curve LaTeX for {task}")

            # K ablation LaTeX
            if k_results:
                latex = build_k_ablation_latex(k_results)
                (output_dir / "k_ablation_latex.tex").write_text(latex)
                print("  Saved k ablation LaTeX")

            # WL depth LaTeX
            if wl_results:
                latex = build_wl_depth_latex(wl_results)
                (output_dir / "wl_depth_latex.tex").write_text(latex)
                print("  Saved WL depth LaTeX")

            # Direction ablation LaTeX
            if dir_results:
                latex = build_direction_ablation_latex(dir_results)
                (output_dir / "direction_ablation_latex.tex").write_text(latex)
                print("  Saved direction ablation LaTeX")

            # Dim control LaTeX
            if dim_results:
                latex = build_dim_control_latex(dim_results)
                (output_dir / "dim_control_latex.tex").write_text(latex)
                print("  Saved dim control LaTeX")

            # Statistical tests LaTeX
            if stat_tests:
                latex = build_statistical_tests_latex(stat_tests)
                (output_dir / "statistical_tests_latex.tex").write_text(latex)
                print("  Saved statistical tests LaTeX")

        print("\nRevision mode complete.")


if __name__ == "__main__":
    main()
