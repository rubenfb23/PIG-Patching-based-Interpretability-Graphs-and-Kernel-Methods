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
