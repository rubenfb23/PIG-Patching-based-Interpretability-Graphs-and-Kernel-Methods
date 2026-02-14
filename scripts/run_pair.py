#!/usr/bin/env python3
"""Run one A/B continual-learning pair experiment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Disable background safetensors conversion checks by default.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
if "--local-files-only" in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clmi.causal.patching import (
    compute_functional_kl_ab_ba,
    compute_intervention_effect_embedding,
)
from clmi.causal.subspaces import (
    build_task_projectors,
    sample_random_projectors,
)
from clmi.data.synth_tasks import TaskPair, generate_task_pair
from clmi.kernels.kernels import k_func, k_nc, k_proj
from clmi.metrics.forgetting import compute_forgetting, compute_hysteresis_metrics
from clmi.metrics.noncommutativity import compute_operator_noncommutativity
from clmi.metrics.robustness import evaluate_robustness
from clmi.model.finetune import (
    compute_anchor_targets,
    evaluate_task_accuracy,
    finetune_on_task,
    save_checkpoint,
)
from clmi.model.gpt2_loader import load_model_tokenizer
from clmi.utils.config import (
    DataConfig,
    ExperimentConfig,
    InterventionConfig,
    MitigationConfig,
    RobustnessConfig,
    SubspaceConfig,
    TrainConfig,
)
from clmi.utils.io import (
    append_csv,
    cache_dir,
    checkpoints_dir,
    run_dir,
    save_csv,
    save_json,
    tables_dir,
)
from clmi.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one CLMI A/B pair.")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-id", type=int, default=0)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--n-keys", type=int, default=400)
    parser.add_argument("--n-values", type=int, default=400)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--bootstrap-iters", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--ft-mode", choices=["full", "lora"], default="full")
    parser.add_argument(
        "--mitigation",
        choices=["none", "freeze_nc", "anchor_reg"],
        default="none",
    )
    parser.add_argument("--top-nc-layers", type=int, default=2)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-steps-a", type=int, default=120)
    parser.add_argument("--max-steps-b", type=int, default=120)
    parser.add_argument("--max-steps-a2", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--robustness-samples", type=int, default=64)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--append-summary", action="store_true")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _make_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    overlap_tag = str(args.overlap).replace(".", "p")
    return (
        f"pair_o{overlap_tag}_s{args.seed}_p{args.pair_id}_"
        f"{args.ft_mode}_{args.mitigation}"
    )


def _first_answer_ids(task_pair: TaskPair) -> tuple[list[int], list[int]]:
    pos_a = [int(ex.answer_token_ids[0]) for ex in task_pair.task_a.examples]
    pos_b = [int(ex.answer_token_ids[0]) for ex in task_pair.task_b.examples]
    return pos_a, pos_b


def _rotated_negatives(token_ids: list[int], vocab_size: int) -> list[int]:
    rotated = token_ids[1:] + token_ids[:1]
    return [
        rid if rid != tid else (rid + 1) % vocab_size
        for tid, rid in zip(token_ids, rotated, strict=True)
    ]


def _build_batch_for_grad(
    examples,
    tokenizer,
    device: str,
):
    pad_id = tokenizer.pad_token_id
    input_rows = []
    label_rows = []

    for ex in examples:
        prompt_ids = tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
        answer_ids = list(ex.answer_token_ids)
        full_ids = prompt_ids + answer_ids
        labels = [-100] * len(full_ids)
        for j, tok in enumerate(answer_ids):
            labels[len(prompt_ids) + j] = tok
        input_rows.append(full_ids)
        label_rows.append(labels)

    max_len = max(len(r) for r in input_rows)
    for i in range(len(input_rows)):
        pad_n = max_len - len(input_rows[i])
        input_rows[i] = input_rows[i] + [pad_id] * pad_n
        label_rows[i] = label_rows[i] + [-100] * pad_n

    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    labels = torch.tensor(label_rows, dtype=torch.long, device=device)
    attention_mask = (input_ids != pad_id).long()
    return input_ids, attention_mask, labels


def _flatten_gradients(model) -> torch.Tensor:
    grads = []
    for param in model.parameters():
        if param.grad is None:
            continue
        grads.append(param.grad.detach().reshape(-1))
    if not grads:
        return torch.zeros(1)
    return torch.cat(grads)


def compute_gradient_overlap_baseline(
    model,
    tokenizer,
    task_pair: TaskPair,
    device: str,
    batch_size: int,
) -> float:
    """Cosine overlap between gradients from A and B on M0."""
    model.train()

    batch_a = list(task_pair.task_a.examples[:batch_size])
    batch_b = list(task_pair.task_b.examples[:batch_size])

    model.zero_grad(set_to_none=True)
    in_a, att_a, lab_a = _build_batch_for_grad(batch_a, tokenizer, device)
    loss_a = model(input_ids=in_a, attention_mask=att_a, labels=lab_a, use_cache=False).loss
    loss_a.backward()
    grad_a = _flatten_gradients(model)

    model.zero_grad(set_to_none=True)
    in_b, att_b, lab_b = _build_batch_for_grad(batch_b, tokenizer, device)
    loss_b = model(input_ids=in_b, attention_mask=att_b, labels=lab_b, use_cache=False).loss
    loss_b.backward()
    grad_b = _flatten_gradients(model)

    model.zero_grad(set_to_none=True)

    denom = torch.norm(grad_a) * torch.norm(grad_b)
    if float(denom) < 1e-12:
        return 0.0
    return float(torch.dot(grad_a, grad_b) / denom)


@torch.no_grad()
def compute_prompt_embedding_similarity(
    model,
    tokenizer,
    task_pair: TaskPair,
    device: str,
    n_samples: int = 64,
) -> float:
    """Surface baseline: cosine similarity of mean prompt embeddings."""
    emb = model.get_input_embeddings()

    def _mean_emb(prompts: list[str]) -> torch.Tensor:
        toks = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        reps = emb(toks["input_ids"])
        mask = toks["attention_mask"].unsqueeze(-1)
        summed = (reps * mask).sum(dim=(0, 1))
        denom = torch.clamp(mask.sum(), min=1.0)
        return summed / denom

    prompts_a = [ex.prompt for ex in task_pair.task_a.examples[:n_samples]]
    prompts_b = [ex.prompt for ex in task_pair.task_b.examples[:n_samples]]
    mean_a = _mean_emb(prompts_a)
    mean_b = _mean_emb(prompts_b)

    denom = torch.norm(mean_a) * torch.norm(mean_b)
    if float(denom) < 1e-12:
        return 0.0
    return float(torch.dot(mean_a, mean_b) / denom)


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.n_keys = min(args.n_keys, 40)
        args.n_values = min(args.n_values, 40)
        args.max_steps_a = min(args.max_steps_a, 12)
        args.max_steps_b = min(args.max_steps_b, 12)
        args.max_steps_a2 = min(args.max_steps_a2, 12)
        args.bootstrap_iters = min(args.bootstrap_iters, 8)
        args.batch_size = min(args.batch_size, 4)
        args.robustness_samples = min(args.robustness_samples, 20)

    run_id = _make_run_id(args)
    seed_everything(args.seed)

    model, tokenizer, device = load_model_tokenizer(
        model_name=args.model,
        device=args.device,
        local_files_only=args.local_files_only,
    )

    pair = generate_task_pair(
        tokenizer,
        overlap=args.overlap,
        n_keys=args.n_keys,
        n_values=args.n_values,
        seed=args.seed,
        prompt_template="Q: {X}\nA:",
        max_answer_tokens=2,
    )

    layers = tuple(range(model.config.n_layer))

    exp_cfg = ExperimentConfig(
        run_id=run_id,
        pair_id=args.pair_id,
        overlap=args.overlap,
        seed=args.seed,
        data=DataConfig(
            n_keys=args.n_keys,
            n_values=args.n_values,
            overlap=args.overlap,
        ),
        train=TrainConfig(
            model_name=args.model,
            device=device,
            seed=args.seed,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_steps=max(args.max_steps_a, args.max_steps_b, args.max_steps_a2),
            eval_every=args.eval_every,
            ft_mode=args.ft_mode,
        ),
        subspace=SubspaceConfig(
            k=args.k,
            bootstrap_iters=args.bootstrap_iters,
            layers=layers,
        ),
        intervention=InterventionConfig(beta=args.beta, mode="reinforce", layers=layers),
        robustness=RobustnessConfig(n_samples=args.robustness_samples),
        mitigation=MitigationConfig(
            mode=args.mitigation,
            top_nc_layers=args.top_nc_layers,
            anchor_weight=args.anchor_weight,
            anchor_samples=args.robustness_samples,
        ),
    )

    run_path = run_dir(run_id)
    ckp_dir = checkpoints_dir(run_id)
    proj_cache = cache_dir("projectors")

    save_json(run_path / "config.json", exp_cfg)
    save_checkpoint(model, tokenizer, ckp_dir, "M0")

    grad_overlap = compute_gradient_overlap_baseline(
        model=model,
        tokenizer=tokenizer,
        task_pair=pair,
        device=device,
        batch_size=min(args.batch_size, 8),
    )
    prompt_embed_sim = compute_prompt_embedding_similarity(
        model=model,
        tokenizer=tokenizer,
        task_pair=pair,
        device=device,
        n_samples=min(64, len(pair.task_a.examples)),
    )

    # Phase A
    result_a = finetune_on_task(
        model=model,
        tokenizer=tokenizer,
        task=pair.task_a,
        train_cfg=exp_cfg.train,
        device=device,
        phase_name="A",
        epochs=1,
        max_steps=args.max_steps_a,
        eval_task=pair.task_a,
    )
    model = result_a.model
    save_checkpoint(model, tokenizer, ckp_dir, "MA")

    acc_a_ma = evaluate_task_accuracy(
        model,
        tokenizer,
        pair.task_a,
        device=device,
        top_k=5,
    )["acc_exact"]

    # Pre-B NC to select protected layers if mitigation is enabled.
    pack_a_pre, _ = build_task_projectors(
        model,
        tokenizer,
        pair.task_a,
        cfg=exp_cfg.subspace,
        device=device,
        cache_root=proj_cache,
        model_name=args.model,
        state_tag=f"{run_id}_MA_A_pre",
    )
    pack_b_pre, _ = build_task_projectors(
        model,
        tokenizer,
        pair.task_b,
        cfg=exp_cfg.subspace,
        device=device,
        cache_root=proj_cache,
        model_name=args.model,
        state_tag=f"{run_id}_MA_B_pre",
    )
    pre_nc = compute_operator_noncommutativity(
        pack_a_pre.projectors,
        pack_b_pre.projectors,
    )
    sorted_layers = sorted(
        pre_nc["nc_layers"].items(), key=lambda x: x[1], reverse=True
    )
    top_nc_layers = [layer for layer, _ in sorted_layers[: args.top_nc_layers]]

    protected_layers = list(top_nc_layers)
    freeze_layers = None
    if args.mitigation == "freeze_nc":
        # In LoRA mode, freezing every block can remove all trainable adapter params.
        # Keep at least one block unfrozen so phase-B optimization remains well-defined.
        if args.ft_mode == "lora":
            n_layers = int(getattr(model.config, "n_layer", len(top_nc_layers)))
            max_protected = max(0, n_layers - 1)
            protected_layers = protected_layers[:max_protected]
        freeze_layers = protected_layers
    anchor_targets = None
    if args.mitigation == "anchor_reg" and top_nc_layers:
        anchor_targets = compute_anchor_targets(
            model,
            tokenizer,
            pair.task_a,
            layers=top_nc_layers,
            n_samples=args.robustness_samples,
            batch_size=args.batch_size,
            device=device,
        )

    # Phase B
    result_b = finetune_on_task(
        model=model,
        tokenizer=tokenizer,
        task=pair.task_b,
        train_cfg=exp_cfg.train,
        device=device,
        phase_name="B",
        epochs=1,
        max_steps=args.max_steps_b,
        freeze_layers=freeze_layers,
        anchor_targets=anchor_targets,
        anchor_weight=args.anchor_weight if anchor_targets else 0.0,
        eval_task=pair.task_b,
    )
    model = result_b.model
    save_checkpoint(model, tokenizer, ckp_dir, "MAB")

    acc_a_mab = evaluate_task_accuracy(
        model,
        tokenizer,
        pair.task_a,
        device=device,
        top_k=5,
    )["acc_exact"]
    acc_b_mab = evaluate_task_accuracy(
        model,
        tokenizer,
        pair.task_b,
        device=device,
        top_k=5,
    )["acc_exact"]

    forgetting_a = compute_forgetting(acc_a_ma, acc_a_mab)

    # Subspaces and non-commutativity on MAB
    pack_a_mab, cache_hit_a = build_task_projectors(
        model,
        tokenizer,
        pair.task_a,
        cfg=exp_cfg.subspace,
        device=device,
        cache_root=proj_cache,
        model_name=args.model,
        state_tag=f"{run_id}_MAB_A",
    )
    pack_b_mab, cache_hit_b = build_task_projectors(
        model,
        tokenizer,
        pair.task_b,
        cfg=exp_cfg.subspace,
        device=device,
        cache_root=proj_cache,
        model_name=args.model,
        state_tag=f"{run_id}_MAB_B",
    )

    nc = compute_operator_noncommutativity(
        pack_a_mab.projectors,
        pack_b_mab.projectors,
    )

    np.savez_compressed(
        run_path / "bases_A.npz",
        **{f"U_{layer}": pack_a_mab.bases[layer] for layer in pack_a_mab.layers},
    )
    np.savez_compressed(
        run_path / "bases_B.npz",
        **{f"U_{layer}": pack_b_mab.bases[layer] for layer in pack_b_mab.layers},
    )

    probe_prompts = [ex.prompt for ex in pair.task_a.examples[: args.robustness_samples]]
    kl_ab_ba = compute_functional_kl_ab_ba(
        model,
        tokenizer,
        prompts=probe_prompts,
        projectors_a=pack_a_mab.projectors,
        projectors_b=pack_b_mab.projectors,
        beta=args.beta,
        mode="reinforce",
        device=device,
    )

    pos_a, pos_b = _first_answer_ids(pair)
    neg_a = _rotated_negatives(pos_a, model.config.vocab_size)
    neg_b = _rotated_negatives(pos_b, model.config.vocab_size)

    phi_a = compute_intervention_effect_embedding(
        model,
        tokenizer,
        prompts=[ex.prompt for ex in pair.task_a.examples[: args.robustness_samples]],
        positive_token_ids=pos_a[: args.robustness_samples],
        negative_token_ids=neg_a[: args.robustness_samples],
        projectors=pack_a_mab.projectors,
        beta=args.beta,
        mode="reinforce",
        device=device,
    )
    phi_b = compute_intervention_effect_embedding(
        model,
        tokenizer,
        prompts=[ex.prompt for ex in pair.task_b.examples[: args.robustness_samples]],
        positive_token_ids=pos_b[: args.robustness_samples],
        negative_token_ids=neg_b[: args.robustness_samples],
        projectors=pack_b_mab.projectors,
        beta=args.beta,
        mode="reinforce",
        device=device,
    )

    k_proj_value = k_proj(pack_a_mab.projectors, pack_b_mab.projectors)
    k_nc_value = k_nc(
        pack_a_mab.projectors,
        pack_b_mab.projectors,
        gamma=args.gamma,
    )
    k_func_value = k_func(phi_a, phi_b, kind="linear", gamma=args.gamma)

    random_proj_a = sample_random_projectors(
        list(layers),
        d_model=model.config.n_embd,
        k=args.k,
        seed=args.seed + 1000,
    )
    random_proj_b = sample_random_projectors(
        list(layers),
        d_model=model.config.n_embd,
        k=args.k,
        seed=args.seed + 2000,
    )
    random_nc = compute_operator_noncommutativity(random_proj_a, random_proj_b)
    random_k_proj = k_proj(random_proj_a, random_proj_b)
    random_k_nc = k_nc(random_proj_a, random_proj_b, gamma=args.gamma)

    robust_a = evaluate_robustness(
        model,
        tokenizer,
        pair.task_a,
        device=device,
        cfg=exp_cfg.robustness,
        seed=args.seed,
    )
    robust_b = evaluate_robustness(
        model,
        tokenizer,
        pair.task_b,
        device=device,
        cfg=exp_cfg.robustness,
        seed=args.seed + 1,
    )

    # Phase A2 for hysteresis
    result_a2 = finetune_on_task(
        model=model,
        tokenizer=tokenizer,
        task=pair.task_a,
        train_cfg=exp_cfg.train,
        device=device,
        phase_name="A2",
        epochs=1,
        max_steps=args.max_steps_a2,
        eval_task=pair.task_a,
    )
    model = result_a2.model
    save_checkpoint(model, tokenizer, ckp_dir, "MABA")

    eval_curve = (
        result_a2.history["eval_acc_exact"].dropna().astype(float).tolist()
        if "eval_acc_exact" in result_a2.history.columns
        else []
    )
    hysteresis_curve = [acc_a_mab] + eval_curve
    hyst = compute_hysteresis_metrics(hysteresis_curve, acc_ref=acc_a_ma, threshold_ratio=0.95)

    pair_metrics = {
        "run_id": run_id,
        "seed": args.seed,
        "pair_id": args.pair_id,
        "overlap": args.overlap,
        "model": args.model,
        "ft_mode": args.ft_mode,
        "mitigation": args.mitigation,
        "AccA_MA": acc_a_ma,
        "AccA_MAB": acc_a_mab,
        "AccB_MAB": acc_b_mab,
        "forgettingA": forgetting_a,
        "remanenceA": hyst["remanence"],
        "coercivity_steps": hyst["coercivity_steps"],
        "hysteresis_area": hyst["hysteresis_area"],
        "NC_global": nc["nc_global"],
        "mean_NC_layers": nc["mean_nc_layers"],
        "KL_AB_BA": kl_ab_ba,
        "k_proj": k_proj_value,
        "k_NC": k_nc_value,
        "k_func": k_func_value,
        "grad_overlap": grad_overlap,
        "prompt_embed_sim": prompt_embed_sim,
        "random_nc_control": random_nc["nc_global"],
        "random_k_proj_control": random_k_proj,
        "random_k_nc_control": random_k_nc,
        "cache_hit_A": int(cache_hit_a),
        "cache_hit_B": int(cache_hit_b),
        "protected_layers": ",".join(str(x) for x in protected_layers),
    }

    pair_path = run_path / "pair_metrics.json"
    save_json(pair_path, pair_metrics)

    nc_layers_df = pd.DataFrame(
        {
            "layer": list(nc["nc_layers"].keys()),
            "NC_l": list(nc["nc_layers"].values()),
            "overlap": args.overlap,
            "seed": args.seed,
            "pair_id": args.pair_id,
        }
    )
    save_csv(run_path / "nc_layers.csv", nc_layers_df)

    curve_df = pd.DataFrame(
        {
            "step_idx": list(range(len(hysteresis_curve))),
            "acc_a": hysteresis_curve,
            "overlap": args.overlap,
            "seed": args.seed,
            "pair_id": args.pair_id,
        }
    )
    save_csv(run_path / "hysteresis_curve.csv", curve_df)

    hist_df = pd.concat([result_a.history, result_b.history, result_a2.history], ignore_index=True)
    save_csv(run_path / "train_history.csv", hist_df)

    save_csv(run_path / "robustness_A.csv", robust_a)
    save_csv(run_path / "robustness_B.csv", robust_b)

    summary_df = pd.DataFrame([pair_metrics])
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else tables_dir() / "per_pair_metrics.csv"
    )

    if args.append_summary:
        append_csv(summary_path, summary_df)
    else:
        save_csv(summary_path, summary_df)

    print(f"[run_pair] completed run_id={run_id}")
    print(pair_metrics)


if __name__ == "__main__":
    main()
