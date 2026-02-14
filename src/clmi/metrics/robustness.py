"""Robustness evaluation under input and embedding perturbations."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from clmi.data.synth_tasks import TaskMapping, XYExample
from clmi.model.finetune import evaluate_task_accuracy
from clmi.utils.config import RobustnessConfig


def _apply_typo(word: str, rng: np.random.Generator) -> str:
    if len(word) < 3:
        return word
    op = rng.choice(["swap", "delete", "replace"])
    chars = list(word)

    if op == "swap" and len(chars) >= 2:
        idx = int(rng.integers(0, len(chars) - 1))
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif op == "delete" and len(chars) >= 2:
        idx = int(rng.integers(0, len(chars)))
        chars.pop(idx)
    else:
        idx = int(rng.integers(0, len(chars)))
        chars[idx] = chr(int(rng.integers(ord("a"), ord("z") + 1)))

    return "".join(chars)


def _extract_key_from_prompt(prompt: str) -> tuple[str, str, str]:
    prefix = "Q: "
    middle = "\nA:"
    if prefix in prompt and middle in prompt:
        start = prompt.index(prefix) + len(prefix)
        end = prompt.index(middle)
        key = prompt[start:end]
        return prompt[:start], key, prompt[end:]
    return "", prompt, ""


def _perturb_task_inputs(
    task: TaskMapping,
    typo_probability: float,
    token_flip_probability: float,
    seed: int,
) -> TaskMapping:
    rng = np.random.default_rng(seed)
    keys = [ex.x for ex in task.examples]

    perturbed: list[XYExample] = []
    for ex in task.examples:
        key = ex.x

        if rng.random() < typo_probability:
            key = _apply_typo(key.strip(), rng)
            if ex.x.startswith(" "):
                key = f" {key}"

        if rng.random() < token_flip_probability:
            key = str(rng.choice(keys))

        pfx, _, sfx = _extract_key_from_prompt(ex.prompt)
        prompt = f"{pfx}{key}{sfx}" if pfx else ex.prompt

        perturbed.append(
            replace(
                ex,
                x=key,
                prompt=prompt,
            )
        )

    return TaskMapping(task_id=f"{task.task_id}_noisy", examples=tuple(perturbed))


@torch.no_grad()
def evaluate_robustness(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task: TaskMapping,
    device: str,
    cfg: RobustnessConfig,
    seed: int,
) -> pd.DataFrame:
    """Evaluate clean vs noisy-input vs noisy-embedding performance."""
    rows: list[dict[str, float | str]] = []

    clean_metrics = evaluate_task_accuracy(
        model,
        tokenizer,
        task,
        device=device,
        top_k=5,
        max_examples=cfg.n_samples,
    )
    rows.append({"scenario": "clean", **clean_metrics})

    noisy_task = _perturb_task_inputs(
        task,
        typo_probability=cfg.typo_probability,
        token_flip_probability=cfg.token_flip_probability,
        seed=seed,
    )
    noisy_input_metrics = evaluate_task_accuracy(
        model,
        tokenizer,
        noisy_task,
        device=device,
        top_k=5,
        max_examples=cfg.n_samples,
    )
    rows.append({"scenario": "input_noise", **noisy_input_metrics})

    emb_module = model.get_input_embeddings()

    def _noise_hook(_module, _inputs, output):
        return output + cfg.embedding_noise_std * torch.randn_like(output)

    handle = emb_module.register_forward_hook(_noise_hook)
    try:
        noisy_emb_metrics = evaluate_task_accuracy(
            model,
            tokenizer,
            task,
            device=device,
            top_k=5,
            max_examples=cfg.n_samples,
        )
    finally:
        handle.remove()

    rows.append({"scenario": "embedding_noise", **noisy_emb_metrics})
    return pd.DataFrame(rows)
