"""Fine-tuning utilities for continual-learning phases (A, B, A2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from clmi.data.synth_tasks import TaskMapping, XYExample
from clmi.utils.config import TrainConfig
from clmi.utils.io import ensure_dir

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:  # pragma: no cover - optional dependency handled at runtime
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None


@dataclass
class FinetuneResult:
    """Result bundle returned by each training phase."""

    model: PreTrainedModel
    history: pd.DataFrame
    total_steps: int


def _base_model(model: PreTrainedModel) -> PreTrainedModel:
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def _prepare_model_for_mode(
    model: PreTrainedModel,
    train_cfg: TrainConfig,
) -> PreTrainedModel:
    if train_cfg.ft_mode == "lora":
        if get_peft_model is None or LoraConfig is None or TaskType is None:
            raise RuntimeError("peft is required for LoRA mode. Install `peft`.")
        if PeftModel is not None and isinstance(model, PeftModel):
            return model

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=train_cfg.lora_r,
            lora_alpha=train_cfg.lora_alpha,
            lora_dropout=train_cfg.lora_dropout,
            target_modules=list(train_cfg.lora_target_modules),
            inference_mode=False,
        )
        model = get_peft_model(model, lora_cfg)
        return model

    for param in model.parameters():
        param.requires_grad = True
    return model


def freeze_transformer_layers(model: PreTrainedModel, layers: Iterable[int]) -> None:
    """Freeze selected transformer blocks by index."""
    base = _base_model(model)
    blocks = getattr(base, "transformer").h
    for layer in layers:
        if layer < 0 or layer >= len(blocks):
            continue
        for param in blocks[layer].parameters():
            param.requires_grad = False


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    tag: str,
) -> Path:
    """Save model/tokenizer checkpoint for a specific phase tag."""
    checkpoint_path = ensure_dir(output_dir / tag)
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    return checkpoint_path


def _build_batch(
    examples: list[XYExample],
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    loss_mask_answer_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer pad_token_id is required.")

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    answer_masks: list[list[int]] = []

    for example in examples:
        prompt_ids = tokenizer(example.prompt, add_special_tokens=False)["input_ids"]
        answer_ids = list(example.answer_token_ids)
        full_ids = prompt_ids + answer_ids

        if loss_mask_answer_only:
            label = [-100] * len(full_ids)
            for j, tok in enumerate(answer_ids):
                label[len(prompt_ids) + j] = tok
        else:
            label = list(full_ids)

        answer_mask = [0] * len(full_ids)
        for j in range(len(answer_ids)):
            answer_mask[len(prompt_ids) + j] = 1

        input_ids.append(full_ids)
        labels.append(label)
        answer_masks.append(answer_mask)

    max_len = max(len(x) for x in input_ids)

    def _pad(xs: list[list[int]], pad_value: int) -> torch.Tensor:
        out = []
        for row in xs:
            out.append(row + [pad_value] * (max_len - len(row)))
        return torch.tensor(out, dtype=torch.long, device=device)

    input_tensor = _pad(input_ids, pad_id)
    label_tensor = _pad(labels, -100)
    answer_mask_tensor = _pad(answer_masks, 0).to(dtype=torch.float32)
    attention_mask = (input_tensor != pad_id).long()

    return input_tensor, attention_mask, label_tensor, answer_mask_tensor


def _iter_batches(
    examples: tuple[XYExample, ...],
    batch_size: int,
    rng: np.random.Generator,
):
    indices = rng.permutation(len(examples))
    shuffled = [examples[int(i)] for i in indices]
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]


@torch.no_grad()
def evaluate_task_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task: TaskMapping,
    device: str,
    top_k: int = 1,
    max_examples: int | None = None,
) -> dict[str, float]:
    """Evaluate token and exact-sequence next-token accuracy on a task."""
    model.eval()
    n = len(task.examples) if max_examples is None else min(max_examples, len(task.examples))

    exact_hits = 0
    token_hits = 0
    topk_hits = 0
    total_tokens = 0

    for example in task.examples[:n]:
        prompt_ids = tokenizer(example.prompt, add_special_tokens=False)["input_ids"]
        answer_ids = list(example.answer_token_ids)
        full_ids = prompt_ids + answer_ids
        input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)

        logits = model(input_ids=input_ids).logits[0]
        all_correct = True

        for j, target_id in enumerate(answer_ids):
            pos = len(prompt_ids) + j - 1
            pred_scores = logits[pos]
            pred_top1 = int(torch.argmax(pred_scores).item())
            pred_topk = torch.topk(pred_scores, k=max(1, top_k)).indices.tolist()

            if pred_top1 == target_id:
                token_hits += 1
            else:
                all_correct = False

            if target_id in pred_topk:
                topk_hits += 1

            total_tokens += 1

        if all_correct:
            exact_hits += 1

    return {
        "acc_exact": exact_hits / max(1, n),
        "acc_token": token_hits / max(1, total_tokens),
        "acc_topk": topk_hits / max(1, total_tokens),
    }


@torch.no_grad()
def compute_anchor_targets(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task: TaskMapping,
    layers: list[int],
    n_samples: int,
    batch_size: int,
    device: str,
) -> dict[int, torch.Tensor]:
    """Compute mean residual activations on answer positions for anchor regularization."""
    if not layers:
        return {}

    model.eval()
    accum: dict[int, torch.Tensor] = {}
    counts: dict[int, torch.Tensor] = {}

    subset = list(task.examples[: min(n_samples, len(task.examples))])
    for start in range(0, len(subset), batch_size):
        batch = subset[start : start + batch_size]
        input_ids, attention_mask, _, answer_mask = _build_batch(
            batch,
            tokenizer,
            device=device,
            loss_mask_answer_only=True,
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        for layer in layers:
            hidden = outputs.hidden_states[layer + 1]
            mask = answer_mask.unsqueeze(-1)
            masked_sum = (hidden * mask).sum(dim=(0, 1))
            masked_count = mask.sum()

            if layer not in accum:
                accum[layer] = masked_sum.detach().clone()
                counts[layer] = masked_count.detach().clone()
            else:
                accum[layer] += masked_sum
                counts[layer] += masked_count

    targets: dict[int, torch.Tensor] = {}
    for layer in layers:
        denom = torch.clamp(counts[layer], min=1.0)
        targets[layer] = (accum[layer] / denom).detach()
    return targets


def finetune_on_task(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task: TaskMapping,
    train_cfg: TrainConfig,
    device: str,
    phase_name: str,
    epochs: int,
    max_steps: int | None = None,
    freeze_layers: list[int] | None = None,
    anchor_targets: dict[int, torch.Tensor] | None = None,
    anchor_weight: float = 0.0,
    eval_task: TaskMapping | None = None,
) -> FinetuneResult:
    """Fine-tune model on one task phase and return step history."""
    model = _prepare_model_for_mode(model, train_cfg)

    if freeze_layers:
        freeze_transformer_layers(model, freeze_layers)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    steps_target = max_steps if max_steps is not None else train_cfg.max_steps
    if steps_target <= 0:
        steps_target = epochs * math.ceil(len(task.examples) / train_cfg.batch_size)
    fixed_steps_mode = max_steps is not None and max_steps > 0

    history_rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(train_cfg.seed)

    model.train()
    step = 0
    epoch = 0
    epoch_limit = max(1, epochs) if not fixed_steps_mode else 1_000_000_000

    pbar = tqdm(total=steps_target, desc=f"train-{phase_name}", leave=False)

    while step < steps_target and epoch < epoch_limit:
        epoch += 1
        for batch in _iter_batches(task.examples, train_cfg.batch_size, rng):
            if step >= steps_target:
                break

            input_ids, attention_mask, labels, answer_mask = _build_batch(
                batch,
                tokenizer,
                device=device,
                loss_mask_answer_only=train_cfg.loss_mask_answer_only,
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=bool(anchor_targets),
                use_cache=False,
            )
            loss = outputs.loss

            if anchor_targets:
                penalty = torch.zeros((), device=device)
                for layer, target in anchor_targets.items():
                    hidden = outputs.hidden_states[layer + 1]
                    mask = answer_mask.unsqueeze(-1)
                    mean_hidden = (hidden * mask).sum(dim=(0, 1)) / torch.clamp(
                        mask.sum(), min=1.0
                    )
                    penalty = penalty + torch.mean((mean_hidden - target.to(device)) ** 2)
                loss = loss + anchor_weight * penalty

            loss.backward()
            clip_grad_norm_(trainable_params, train_cfg.gradient_clip_norm)
            optimizer.step()

            step += 1
            pbar.update(1)

            row: dict[str, float | int | str] = {
                "phase": phase_name,
                "step": step,
                "loss": float(loss.detach().item()),
            }

            if eval_task is not None and (step % train_cfg.eval_every == 0 or step == steps_target):
                eval_metrics = evaluate_task_accuracy(
                    model,
                    tokenizer,
                    eval_task,
                    device=device,
                    top_k=5,
                    max_examples=128,
                )
                row.update(
                    {
                        "eval_acc_exact": eval_metrics["acc_exact"],
                        "eval_acc_token": eval_metrics["acc_token"],
                    }
                )

            history_rows.append(row)

    pbar.close()
    history = pd.DataFrame(history_rows)
    return FinetuneResult(model=model, history=history, total_steps=step)
