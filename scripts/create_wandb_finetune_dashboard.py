#!/usr/bin/env python3
"""Create a W&B dashboard focused on GPT-2 fine-tuning diagnostics."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a W&B fine-tuning QA dashboard/workspace.",
    )
    parser.add_argument(
        "--entity",
        default="",
        help="W&B entity (team/user). Defaults to logged-in default entity.",
    )
    parser.add_argument(
        "--project",
        default="pig-finetune",
        help="W&B project that contains finetune runs.",
    )
    parser.add_argument(
        "--name",
        default="PIG Finetune QA Dashboard",
        help="Workspace display name.",
    )
    parser.add_argument(
        "--query",
        default="tags:finetune",
        help='Runset query filter (example: \'tags:finetune config.model_name:"gpt2"\').',
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=20,
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
        raise RuntimeError("Could not resolve W&B entity. Pass --entity explicitly.")

    workspace = ws.Workspace(
        entity=entity,
        project=args.project,
        name=args.name,
        runset_settings=ws.RunsetSettings(query=args.query),
        settings=ws.WorkspaceSettings(
            x_axis="Step",
            smoothing_type="none",
            max_runs=max(1, args.max_runs),
            ignore_outliers=True,
        ),
        auto_generate_panels=False,
        sections=[
            ws.Section(
                name="Train Dynamics",
                pinned=True,
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Train Loss",
                        x="train/global_step",
                        y=["train/loss"],
                    ),
                    wr.LinePlot(
                        title="Learning Rate",
                        x="train/global_step",
                        y=["train/lr"],
                    ),
                    wr.LinePlot(
                        title="Throughput (tokens/sec)",
                        x="train/global_step",
                        y=["train/tokens_per_sec"],
                    ),
                ],
            ),
            ws.Section(
                name="Validation",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Validation Loss vs Best",
                        x="epoch",
                        y=["val/loss", "val/best_loss_so_far"],
                    ),
                    wr.LinePlot(
                        title="Validation PPL",
                        x="epoch",
                        y=["val/ppl"],
                    ),
                ],
            ),
            ws.Section(
                name="Base vs Distilled (Test)",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Test Loss: Base vs Distilled",
                        x="epoch",
                        y=["test/base_loss", "test/distilled_loss"],
                    ),
                    wr.LinePlot(
                        title="Test PPL: Base vs Distilled",
                        x="epoch",
                        y=["test/base_ppl", "test/distilled_ppl"],
                    ),
                    wr.LinePlot(
                        title="Delta vs Base",
                        x="epoch",
                        y=["test/delta_loss_vs_base", "test/delta_ppl_vs_base"],
                    ),
                    wr.LinePlot(
                        title="Improvement % vs Base",
                        x="epoch",
                        y=["test/improvement_pct_vs_base"],
                    ),
                ],
            ),
            ws.Section(
                name="Run Outcome",
                is_open=True,
                panels=[
                    wr.ScalarChart(
                        title="Final Distilled Loss",
                        metric="test/distilled_loss",
                        groupby_aggfunc="mean",
                    ),
                    wr.ScalarChart(
                        title="Final Delta Loss vs Base",
                        metric="test/delta_loss_vs_base",
                        groupby_aggfunc="mean",
                    ),
                    wr.ScalarChart(
                        title="Final Improvement % vs Base",
                        metric="test/improvement_pct_vs_base",
                        groupby_aggfunc="mean",
                    ),
                    wr.ScalarChart(
                        title="Final Validation Loss",
                        metric="val/loss",
                        groupby_aggfunc="mean",
                    ),
                ],
            ),
            ws.Section(
                name="System Context",
                is_open=False,
                panels=[
                    wr.LinePlot(
                        title="GPU Utilization",
                        y=[
                            "system/gpu.0.gpu",
                            "system/gpu.1.gpu",
                            "system/gpu.2.gpu",
                            "system/gpu.3.gpu",
                        ],
                    ),
                    wr.LinePlot(
                        title="GPU Memory",
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
