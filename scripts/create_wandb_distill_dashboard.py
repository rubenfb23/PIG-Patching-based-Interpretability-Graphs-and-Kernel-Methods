#!/usr/bin/env python3
"""Create a W&B dashboard focused on distillation quality diagnostics.

This dashboard is meant to answer:
    "Is this dataset being distilled correctly with this teacher model?"
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a W&B distillation QA dashboard/workspace.",
    )
    parser.add_argument(
        "--entity",
        default="",
        help="W&B entity (team/user). Defaults to logged-in default entity.",
    )
    parser.add_argument(
        "--project",
        default="pig-distill",
        help="W&B project that contains distillation runs.",
    )
    parser.add_argument(
        "--name",
        default="PIG Distillation QA Dashboard",
        help="Workspace display name.",
    )
    parser.add_argument(
        "--query",
        default="tags:distill",
        help='Runset query filter (example: \'tags:distill config.teacher_model:"openai/gpt-oss-20b"\').',
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=30,
        help="Maximum runs to overlay in line charts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import wandb
        from wandb.apis import reports as wr
        from wandb.apis import workspaces as ws
    except Exception as exc:
        raise RuntimeError(
            "Missing W&B workspace dependencies. Install with: uv pip install 'wandb[workspaces]'"
        ) from exc

    api = wandb.Api()
    entity = args.entity or api.default_entity
    if not entity:
        raise RuntimeError(
            "Could not resolve W&B entity. Pass --entity explicitly."
        )

    workspace = ws.Workspace(
        entity=entity,
        project=args.project,
        name=args.name,
        runset_settings=ws.RunsetSettings(query=args.query),
        settings=ws.WorkspaceSettings(
            x_axis="attempted",
            smoothing_type="none",
            max_runs=max(1, args.max_runs),
            ignore_outliers=True,
        ),
        auto_generate_panels=False,
        sections=[
            ws.Section(
                name="Fail-Fast Signals",
                pinned=True,
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Parse / Keep / Accuracy",
                        x="attempted",
                        y=[
                            "distill/parse_rate",
                            "distill/keep_rate",
                            "distill/accuracy_estimate",
                        ],
                    ),
                    wr.LinePlot(
                        title="Counts",
                        x="attempted",
                        y=[
                            "distill/parsed_final",
                            "distill/correct",
                            "distill/kept",
                        ],
                    ),
                    wr.LinePlot(
                        title="Throughput (examples/sec)",
                        x="attempted",
                        y=["distill/examples_per_sec"],
                    ),
                    wr.LinePlot(
                        title="Progress",
                        x="attempted",
                        y=["attempted"],
                    ),
                ],
            ),
            ws.Section(
                name="Run Outcome",
                is_open=True,
                panels=[
                    wr.ScalarChart(
                        title="Final Keep Rate",
                        metric="distill/keep_rate",
                        groupby_aggfunc="mean",
                    ),
                    wr.ScalarChart(
                        title="Final Parse Rate",
                        metric="distill/parse_rate",
                        groupby_aggfunc="mean",
                    ),
                    wr.ScalarChart(
                        title="Final Accuracy Estimate",
                        metric="distill/accuracy_estimate",
                        groupby_aggfunc="mean",
                    ),
                    wr.ScalarChart(
                        title="Final Throughput",
                        metric="distill/examples_per_sec",
                        groupby_aggfunc="mean",
                    ),
                ],
            ),
            ws.Section(
                name="Quality vs Speed",
                is_open=True,
                panels=[
                    wr.ScatterPlot(
                        title="Accuracy vs Throughput",
                        x="distill/examples_per_sec",
                        y="distill/accuracy_estimate",
                    ),
                    wr.ScatterPlot(
                        title="Parse Rate vs Keep Rate",
                        x="distill/parse_rate",
                        y="distill/keep_rate",
                    ),
                ],
            ),
            ws.Section(
                name="System Context",
                is_open=False,
                panels=[
                    wr.LinePlot(
                        title="GPU Utilization",
                        x="attempted",
                        y=["system/gpu.0.gpu", "system/gpu.1.gpu", "system/gpu.2.gpu", "system/gpu.3.gpu"],
                    ),
                    wr.LinePlot(
                        title="GPU Memory",
                        x="attempted",
                        y=[
                            "system/gpu.0.memoryAllocated",
                            "system/gpu.1.memoryAllocated",
                            "system/gpu.2.memoryAllocated",
                            "system/gpu.3.memoryAllocated",
                        ],
                    ),
                ],
            ),
        ],
    )

    workspace.save()
    print(workspace.url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
