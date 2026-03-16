#!/usr/bin/env python3
"""Summarize a classical publication study directory."""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path


def _read_summary_rows(study_dir: Path) -> list[dict[str, str]]:
    summary_path = study_dir / "summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_stats(rows: list[dict[str, str]], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    return st.mean(values), st.pstdev(values)


def _slice_balance(study_dir: Path) -> dict[str, list[int]]:
    balances: dict[str, list[int]] = {}
    runs_dir = study_dir / "runs"
    for path in sorted(runs_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        strata = data["causal_eval"]["split"]["strata_counts"]
        for slice_label, counts in strata.items():
            balances.setdefault(slice_label, []).append(int(counts["total"]))
    return balances


def _filter_balance(study_dir: Path) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {
        "overall_retention_rate": [],
        "after_filter_max_to_min_ratio": [],
    }
    runs_dir = study_dir / "runs"
    for path in sorted(runs_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        report = data.get("causal_eval", {}).get("clean_better_filter_report")
        if report:
            metrics["overall_retention_rate"].append(float(report["retention_rate"]))
            ratio = report["balance"]["after_filter"]["max_to_min_ratio"]
            if ratio is not None:
                metrics["after_filter_max_to_min_ratio"].append(float(ratio))
            for slice_label, slice_metrics in report["slice_retention"].items():
                metrics.setdefault(f"{slice_label}::retention_rate", []).append(
                    float(slice_metrics["retention_rate"])
                )
            continue

        corruptions = data.get("corruptions", [])
        num_examples = int(data.get("num_examples_per_corruption", 0))
        split_counts = data.get("causal_eval", {}).get("split", {}).get("strata_counts", {})
        expected_by_slice = {f"ioi:{corruption}": num_examples for corruption in corruptions}
        retained_counts = {
            slice_label: int(counts["total"])
            for slice_label, counts in split_counts.items()
        }
        total_expected = sum(expected_by_slice.values())
        total_retained = sum(retained_counts.values())
        if total_expected > 0:
            metrics["overall_retention_rate"].append(total_retained / total_expected)
        nonzero_retained = [count for count in retained_counts.values() if count > 0]
        if nonzero_retained:
            metrics["after_filter_max_to_min_ratio"].append(
                max(nonzero_retained) / min(nonzero_retained)
            )
        for slice_label, expected in expected_by_slice.items():
            retained = retained_counts.get(slice_label, 0)
            if expected > 0:
                metrics.setdefault(f"{slice_label}::retention_rate", []).append(
                    retained / expected
                )
    return metrics


def _format_line(label: str, mean: float, std: float) -> str:
    return f"- {label}: mean={mean:.4f}, std={std:.4f}"


def build_report(study_dir: Path) -> str:
    rows = _read_summary_rows(study_dir)
    balances = _slice_balance(study_dir)
    filter_metrics = _filter_balance(study_dir)
    lines = [f"# Study Summary: {study_dir}"]
    lines.append("")
    lines.append(f"- runs: {len(rows)}")
    filter_modes = sorted({row.get("causal_filter_mode", "clean_better") for row in rows})
    lines.append(f"- causal_filter_mode: {', '.join(filter_modes)}")
    lines.append(_format_line("linear_accuracy_mean", *_metric_stats(rows, "linear_accuracy_mean")))
    lines.append(_format_line("rbf_accuracy_mean", *_metric_stats(rows, "rbf_accuracy_mean")))
    lines.append(
        _format_line(
            "causal_i_survival_rate_fdr_bh",
            *_metric_stats(rows, "causal_i_survival_rate_fdr_bh"),
        )
    )
    lines.append(
        _format_line(
            "causal_i_significant_fdr_bh",
            *_metric_stats(rows, "causal_i_significant_fdr_bh"),
        )
    )
    lines.append(
        _format_line(
            "causal_num_edges_evaluated",
            *_metric_stats(rows, "causal_num_edges_evaluated"),
        )
    )
    lines.append("")
    lines.append("## Post-filter Slice Balance")
    for slice_label, values in sorted(balances.items()):
        lines.append(
            f"- {slice_label}: mean={st.mean(values):.4f}, min={min(values)}, max={max(values)}, values={values}"
        )
    if filter_metrics:
        lines.append("")
        lines.append("## Filter Retention")
        for key, values in sorted(filter_metrics.items()):
            if not values:
                continue
            lines.append(
                f"- {key}: mean={st.mean(values):.4f}, min={min(values):.4f}, max={max(values):.4f}, values={[round(v, 4) for v in values]}"
            )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summarize_classical_publication_study",
        description="Summarize summary.csv and per-run causal slice balance.",
    )
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(args.study_dir)
    if args.output is None:
        print(report)
        return
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote summary to {args.output}")


if __name__ == "__main__":
    main()
