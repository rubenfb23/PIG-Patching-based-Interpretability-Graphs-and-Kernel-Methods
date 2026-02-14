#!/usr/bin/env python3
"""Run the full CLMI experiment battery across overlaps/seeds/pairs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clmi.kernels.curriculum import greedy_curriculum
from clmi.kernels.kernels import k_func, k_nc, k_proj
from clmi.kernels.predict import fit_kernel_ridge_predictor
from clmi.metrics.statistics import compute_correlation_table
from clmi.utils.io import figures_dir, save_csv, save_json, tables_dir
from clmi.viz.plots import (
    plot_correlation_matrix,
    plot_full_loop,
    plot_hysteresis_curves,
    plot_kernel_matrix,
    plot_mitigation_bars,
    plot_nc_forgetting_scatter,
    plot_nc_layer_heatmap,
    plot_scatter_with_ci,
    plot_weight_distance_bars,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLMI experiment suite.")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--overlaps", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--n_pairs_per_overlap", type=int, default=10)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--ft-modes", nargs="+", default=["full", "lora"])
    parser.add_argument(
        "--mitigations",
        nargs="+",
        default=["none", "freeze_nc", "anchor_reg"],
    )
    parser.add_argument("--n-keys", type=int, default=400)
    parser.add_argument("--n-values", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--learning-rate-lora",
        type=float,
        default=None,
        help="Override learning rate for LoRA runs (default: 10x --learning-rate).",
    )
    parser.add_argument("--max-steps-a", type=int, default=120)
    parser.add_argument("--max-steps-b", type=int, default=120)
    parser.add_argument("--max-steps-a2", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["ABA"],
        help="Task-ordering protocols to test (ABA, BAB).",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results/tables/summary.csv and skip completed run_ids.",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Number of GPUs available. Runs are distributed round-robin "
             "across CUDA devices 0..n-1 in parallel.",
    )
    return parser.parse_args()


def _run_id(
    overlap: float,
    seed: int,
    pair_id: int,
    ft_mode: str,
    mitigation: str,
    protocol: str = "ABA",
) -> str:
    proto_tag = f"_{protocol}" if protocol != "ABA" else ""
    return (
        f"pair_o{str(overlap).replace('.', 'p')}_s{seed}_p{pair_id}_"
        f"{ft_mode}_{mitigation}{proto_tag}"
    )


def _build_run_pair_cmd(args: argparse.Namespace, overlap: float, seed: int, pair_id: int, ft_mode: str, mitigation: str, summary_path: Path, protocol: str = "ABA") -> list[str]:
    run_id = _run_id(overlap, seed, pair_id, ft_mode, mitigation, protocol)
    cmd = [
        sys.executable,
        "scripts/run_pair.py",
        "--model",
        args.model,
        "--device",
        "cuda",  # GPU id is set via CUDA_VISIBLE_DEVICES
        "--seed",
        str(seed),
        "--pair-id",
        str(pair_id),
        "--overlap",
        str(overlap),
        "--n-keys",
        str(args.n_keys),
        "--n-values",
        str(args.n_values),
        "--k",
        str(args.k),
        "--beta",
        str(args.beta),
        "--gamma",
        str(args.gamma),
        "--ft-mode",
        ft_mode,
        "--mitigation",
        mitigation,
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--max-steps-a",
        str(args.max_steps_a),
        "--max-steps-b",
        str(args.max_steps_b),
        "--max-steps-a2",
        str(args.max_steps_a2),
        "--eval-every",
        str(args.eval_every),
        "--run-id",
        run_id,
        "--append-summary",
        "--summary-path",
        str(summary_path),
        "--protocol",
        protocol,
    ]
    if args.learning_rate_lora is not None:
        cmd += ["--learning-rate-lora", str(args.learning_rate_lora)]
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.smoke:
        cmd.append("--smoke")
    return cmd


# --- Multi-GPU parallel dispatch helpers ---

_summary_lock = threading.Lock()


def _run_one(cmd: list[str], gpu_id: int, run_idx: int, total: int, run_id: str, local_files_only: bool) -> tuple[str, int]:
    """Execute a single run_pair.py command on a specific GPU.

    Returns (run_id, returncode).
    """
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if local_files_only:
        child_env.setdefault("HF_HUB_OFFLINE", "1")
        child_env.setdefault("TRANSFORMERS_OFFLINE", "1")
        child_env.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    print(f"[{run_idx}/{total}] GPU {gpu_id} | {run_id}")
    proc = subprocess.run(cmd, check=False, env=child_env)
    if proc.returncode == 0:
        print(f"[run_pair] completed run_id={run_id}")
    else:
        print(f"[run_pair] FAILED run_id={run_id} (exit {proc.returncode})")
    return run_id, proc.returncode


def _load_projectors_from_run(run_id: str, task_label: str) -> dict[int, np.ndarray] | None:
    path = Path("results") / "runs" / run_id / f"bases_{task_label}.npz"
    if not path.exists():
        return None

    data = np.load(path)
    projectors: dict[int, np.ndarray] = {}
    for key in data.files:
        if not key.startswith("U_"):
            continue
        layer = int(key.split("_")[1])
        U = data[key]
        projectors[layer] = U @ U.T
    return projectors


def _build_task_kernel_matrices(summary: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build K_proj, K_nc, and K_func matrices from saved run artifacts.

    Returns (K_proj, K_nc, K_func, labels).
    """
    labels: list[str] = []
    projector_bank: list[dict[int, np.ndarray]] = []
    phi_bank: list[np.ndarray | None] = []

    for _, row in summary.iterrows():
        run_id = str(row["run_id"])
        proj = _load_projectors_from_run(run_id, "A")
        if proj is None:
            continue
        labels.append(run_id)
        projector_bank.append(proj)

        # Load phi embeddings if available
        phi_path = Path("results") / "runs" / run_id / "phi_embeddings.npz"
        if phi_path.exists():
            data = np.load(phi_path)
            phi_bank.append(data["phi_a"])
        else:
            phi_bank.append(None)

    n = len(projector_bank)
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)), labels

    K_proj = np.zeros((n, n), dtype=np.float32)
    K_nc = np.zeros((n, n), dtype=np.float32)
    K_func = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            kp = k_proj(projector_bank[i], projector_bank[j])
            kn = k_nc(projector_bank[i], projector_bank[j], gamma=0.1)
            K_proj[i, j] = kp
            K_proj[j, i] = kp
            K_nc[i, j] = kn
            K_nc[j, i] = kn

            # k_func from phi embeddings
            if phi_bank[i] is not None and phi_bank[j] is not None:
                kf = k_func(phi_bank[i], phi_bank[j], kind="linear", gamma=0.1)
                K_func[i, j] = kf
                K_func[j, i] = kf

    return K_proj, K_nc, K_func, labels


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.seeds = min(args.seeds, 1)
        args.overlaps = [args.overlaps[0]]
        args.n_pairs_per_overlap = min(args.n_pairs_per_overlap, 1)
        args.ft_modes = [args.ft_modes[0]]
        args.mitigations = [args.mitigations[0]]

    summary_path = tables_dir() / "summary.csv"
    completed_run_ids: set[str] = set()
    if summary_path.exists():
        if args.resume:
            existing = pd.read_csv(summary_path)
            if "run_id" in existing.columns:
                completed_run_ids = set(existing["run_id"].astype(str).tolist())
        else:
            summary_path.unlink()

    failures: list[tuple[str, int]] = []
    total_runs = (
        args.seeds
        * len(args.overlaps)
        * args.n_pairs_per_overlap
        * len(args.ft_modes)
        * len(args.mitigations)
        * len(args.protocols)
    )

    run_idx = 0
    # Build the full list of pending jobs first
    pending_jobs: list[tuple[int, str, list[str]]] = []
    for protocol in args.protocols:
        for ft_mode in args.ft_modes:
            for mitigation in args.mitigations:
                for overlap in args.overlaps:
                    for seed in range(args.seeds):
                        for pair_id in range(args.n_pairs_per_overlap):
                            run_idx += 1
                            run_id = _run_id(overlap, seed, pair_id, ft_mode, mitigation, protocol)
                            if run_id in completed_run_ids:
                                print(f"[{run_idx}/{total_runs}] skip {run_id} (already in summary)")
                                continue
                            cmd = _build_run_pair_cmd(
                                args,
                                overlap=overlap,
                                seed=seed,
                                pair_id=pair_id,
                                ft_mode=ft_mode,
                                mitigation=mitigation,
                                summary_path=summary_path,
                                protocol=protocol,
                            )
                            pending_jobs.append((run_idx, run_id, cmd))

    n_gpus = max(1, args.n_gpus)
    print(f"\n=== {len(pending_jobs)} runs pending, dispatching across {n_gpus} GPU(s) ===\n")

    if n_gpus == 1:
        # Sequential (original behavior)
        for job_idx, run_id, cmd in pending_jobs:
            child_env = os.environ.copy()
            if args.local_files_only:
                child_env.setdefault("HF_HUB_OFFLINE", "1")
                child_env.setdefault("TRANSFORMERS_OFFLINE", "1")
                child_env.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
            print(f"[{job_idx}/{total_runs}] {' '.join(cmd)}")
            proc = subprocess.run(cmd, check=False, env=child_env)
            if proc.returncode != 0:
                failures.append((" ".join(cmd), proc.returncode))
                if not args.continue_on_error:
                    raise SystemExit(proc.returncode)
            else:
                completed_run_ids.add(run_id)
    else:
        # Parallel: dispatch across GPUs with a thread pool
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = {}
            for i, (job_idx, run_id, cmd) in enumerate(pending_jobs):
                gpu_id = i % n_gpus
                fut = pool.submit(
                    _run_one, cmd, gpu_id, job_idx, total_runs,
                    run_id, args.local_files_only,
                )
                futures[fut] = (run_id, cmd)

            for fut in as_completed(futures):
                run_id, cmd = futures[fut]
                rid, rc = fut.result()
                if rc != 0:
                    failures.append((" ".join(cmd), rc))
                    if not args.continue_on_error:
                        # Cancel remaining
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise SystemExit(rc)
                else:
                    completed_run_ids.add(rid)

    if not summary_path.exists():
        raise SystemExit("No summary produced. All runs failed?")

    summary = pd.read_csv(summary_path)
    save_csv(tables_dir() / "summary.csv", summary)

    # Aggregate per-layer NC across runs.
    layer_rows: list[pd.DataFrame] = []
    curves: dict[float, list[list[float]]] = {}
    full_loop_dfs: dict[float, list[pd.DataFrame]] = {}
    robustness_rows: list[pd.DataFrame] = []
    train_history_rows: list[pd.DataFrame] = []

    for _, row in summary.iterrows():
        run_id = str(row["run_id"])
        run_path = Path("results") / "runs" / run_id

        nc_layer_path = run_path / "nc_layers.csv"
        if nc_layer_path.exists():
            layer_rows.append(pd.read_csv(nc_layer_path))

        curve_path = run_path / "hysteresis_curve.csv"
        if curve_path.exists():
            curve_df = pd.read_csv(curve_path)
            overlap = float(row["overlap"])
            curves.setdefault(overlap, []).append(curve_df["acc_a"].astype(float).tolist())

        loop_path = run_path / "full_loop_curve.csv"
        if loop_path.exists():
            overlap = float(row["overlap"])
            full_loop_dfs.setdefault(overlap, []).append(pd.read_csv(loop_path))

        for label in ("A", "B"):
            rob_path = run_path / f"robustness_{label}.csv"
            if rob_path.exists():
                rdf = pd.read_csv(rob_path)
                rdf["run_id"] = run_id
                rdf["task"] = label
                robustness_rows.append(rdf)

        hist_path = run_path / "train_history.csv"
        if hist_path.exists():
            hdf = pd.read_csv(hist_path)
            hdf["run_id"] = run_id
            train_history_rows.append(hdf)

    fig_dir = figures_dir()

    # --- Scatter plots ---
    plot_nc_forgetting_scatter(summary, fig_dir / "scatter_nc_vs_forgetting.png")
    # NC vs hysteresis area
    if "hysteresis_area" in summary.columns:
        plot_scatter_with_ci(
            summary, "NC_global", "hysteresis_area",
            fig_dir / "scatter_nc_vs_hysteresis_area.png",
            title="NC_global vs Hysteresis area",
        )
    # NC vs coercivity
    if "coercivity_steps" in summary.columns:
        plot_scatter_with_ci(
            summary, "NC_global", "coercivity_steps",
            fig_dir / "scatter_nc_vs_coercivity.png",
            title="NC_global vs Coercivity (steps)",
        )
    # Weight distance vs forgetting
    if "wdist_MA_MAB_l2" in summary.columns:
        plot_scatter_with_ci(
            summary, "wdist_MA_MAB_l2", "forgettingA",
            fig_dir / "scatter_wdist_vs_forgetting.png",
            title="Weight-distance (MA→MAB) vs Forgetting",
        )
    # Cycle closure distance vs hysteresis
    if "wdist_M0_MABA_l2" in summary.columns and "hysteresis_area" in summary.columns:
        plot_scatter_with_ci(
            summary, "wdist_M0_MABA_l2", "hysteresis_area",
            fig_dir / "scatter_cycle_closure_vs_hysteresis.png",
            title="Cycle-closure distance (M0→MABA) vs Hysteresis area",
        )

    # --- Correlation matrix ---
    corr_cols = [
        "NC_global", "mean_NC_layers", "forgettingA", "forgettingB_after_A2",
        "hysteresis_area", "coercivity_steps", "remanenceA",
        "k_proj", "k_NC", "k_func", "KL_AB_BA",
        "grad_overlap", "prompt_embed_sim",
        "wdist_M0_MA_l2", "wdist_MA_MAB_l2", "wdist_MAB_MABA_l2", "wdist_M0_MABA_l2",
    ]
    plot_correlation_matrix(
        summary, corr_cols, fig_dir / "correlation_matrix_spearman.png",
        method="spearman", title="Spearman correlations between metrics",
    )

    # --- Statistical correlation table ---
    predictors = [
        "NC_global", "k_proj", "k_NC", "k_func", "KL_AB_BA",
        "grad_overlap", "prompt_embed_sim",
        "wdist_MA_MAB_l2", "wdist_M0_MABA_l2",
    ]
    for target in ("forgettingA", "hysteresis_area"):
        if target in summary.columns:
            corr_table = compute_correlation_table(summary, predictors, target)
            if corr_table is not None:
                save_csv(tables_dir() / f"correlations_{target}.csv", corr_table)

    # --- Weight distance bars ---
    plot_weight_distance_bars(summary, fig_dir / "weight_distance_bars.png")

    # --- Layer NC heatmap ---
    if layer_rows:
        layer_df = pd.concat(layer_rows, ignore_index=True)
        save_csv(tables_dir() / "nc_layers_all.csv", layer_df)
        plot_nc_layer_heatmap(
            layer_df,
            fig_dir / "heatmap_nc_per_layer_by_overlap.png",
        )

    # --- Hysteresis curves (A2-only) ---
    plot_hysteresis_curves(curves, fig_dir)

    # --- Full-loop plots (Acc_A + Acc_B across all 3 phases) ---
    for overlap, dfs in full_loop_dfs.items():
        if dfs:
            plot_full_loop(
                dfs,
                fig_dir / f"full_loop_overlap_{overlap}.png",
                title=f"Full A→B→A loop (overlap={overlap})",
            )

    # --- Aggregate robustness ---
    if robustness_rows:
        all_robustness = pd.concat(robustness_rows, ignore_index=True)
        save_csv(tables_dir() / "robustness_all.csv", all_robustness)

    # --- Aggregate training history ---
    if train_history_rows:
        all_history = pd.concat(train_history_rows, ignore_index=True)
        save_csv(tables_dir() / "train_history_all.csv", all_history)

    # --- Kernel matrices ---
    K_proj_mat, K_nc_mat, K_func_mat, labels = _build_task_kernel_matrices(summary)
    if K_proj_mat.size > 0:
        plot_kernel_matrix(
            K_proj_mat,
            labels,
            fig_dir / "kernel_matrix_kproj.png",
            title="Kernel matrix: k_proj",
        )
        plot_kernel_matrix(
            K_nc_mat,
            labels,
            fig_dir / "kernel_matrix_knc.png",
            title="Kernel matrix: k_NC",
        )
        if K_func_mat.any():
            plot_kernel_matrix(
                K_func_mat,
                labels,
                fig_dir / "kernel_matrix_kfunc.png",
                title="Kernel matrix: k_func",
            )

        summary_by_run = summary.drop_duplicates(subset=["run_id"]).set_index("run_id")
        aligned_targets = np.asarray(
            [float(summary_by_run.loc[label, "forgettingA"]) for label in labels],
            dtype=np.float32,
        )

        # Kernel ridge prediction with k_NC
        pred = fit_kernel_ridge_predictor(
            K_nc_mat,
            aligned_targets,
            alpha=1.0,
            cv_folds=min(5, len(labels)),
        )
        pred_df = pd.DataFrame(
            {
                "run_id": labels,
                "forgetting_target": aligned_targets,
                "forgetting_pred_train": pred["pred_train"],
            }
        )
        if "pred_cv" in pred:
            pred_df["forgetting_pred_cv"] = pred["pred_cv"]
        save_csv(tables_dir() / "kernel_predict_forgetting_knc.csv", pred_df)
        save_json(
            tables_dir() / "kernel_predict_metrics_knc.json",
            {
                "kernel": "k_NC",
                "train_rmse": pred["train_rmse"],
                "train_r2": pred["train_r2"],
                "cv_rmse": pred.get("cv_rmse"),
                "cv_r2": pred.get("cv_r2"),
            },
        )

        # Kernel ridge prediction with k_proj
        pred_proj = fit_kernel_ridge_predictor(
            K_proj_mat,
            aligned_targets,
            alpha=1.0,
            cv_folds=min(5, len(labels)),
        )
        save_json(
            tables_dir() / "kernel_predict_metrics_kproj.json",
            {
                "kernel": "k_proj",
                "train_rmse": pred_proj["train_rmse"],
                "train_r2": pred_proj["train_r2"],
                "cv_rmse": pred_proj.get("cv_rmse"),
                "cv_r2": pred_proj.get("cv_r2"),
            },
        )

        # Kernel ridge prediction with k_func (if available)
        if K_func_mat.any():
            pred_func = fit_kernel_ridge_predictor(
                K_func_mat,
                aligned_targets,
                alpha=1.0,
                cv_folds=min(5, len(labels)),
            )
            save_json(
                tables_dir() / "kernel_predict_metrics_kfunc.json",
                {
                    "kernel": "k_func",
                    "train_rmse": pred_func["train_rmse"],
                    "train_r2": pred_func["train_r2"],
                    "cv_rmse": pred_func.get("cv_rmse"),
                    "cv_r2": pred_func.get("cv_r2"),
                },
            )

        curriculum = greedy_curriculum(labels, K_nc_mat, objective="min_conflict")
        curriculum_df = pd.DataFrame(
            {
                "position": list(range(len(curriculum))),
                "run_id": curriculum,
            }
        )
        save_csv(tables_dir() / "curriculum_knc.csv", curriculum_df)

        if len(labels) >= 2:
            n_clusters = min(4, len(labels))
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                random_state=0,
                assign_labels="kmeans",
            )
            clusters = clustering.fit_predict(K_nc_mat)
            cluster_df = pd.DataFrame({"run_id": labels, "cluster": clusters})
            save_csv(tables_dir() / "task_clusters_knc.csv", cluster_df)

    plot_mitigation_bars(summary, fig_dir / "mitigation_comparison_bars.png")

    if failures:
        print("Completed with failures:")
        for cmd, code in failures:
            print(f"- exit={code}: {cmd}")
    else:
        print("All runs completed successfully.")


if __name__ == "__main__":
    main()
