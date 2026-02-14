"""Synthetic A/B dictionary task generation with controlled overlap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from transformers import PreTrainedTokenizerBase

from .token_utils import ensure_answer_token_length, select_clean_tokens


@dataclass(frozen=True)
class XYExample:
    """One prompt-target training/evaluation sample."""

    x: str
    y: str
    prompt: str
    answer_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class TaskMapping:
    """A synthetic dictionary-like task."""

    task_id: str
    examples: tuple[XYExample, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(ex.x for ex in self.examples)

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(ex.y for ex in self.examples)


@dataclass(frozen=True)
class TaskPair:
    """Pair of tasks with known conflicting overlap."""

    task_a: TaskMapping
    task_b: TaskMapping
    shared_keys: tuple[str, ...]
    conflict_keys: tuple[str, ...]


def _build_task(
    task_id: str,
    keys: list[str],
    values: list[str],
    tokenizer: PreTrainedTokenizerBase,
    prompt_template: str,
    max_answer_tokens: int,
) -> TaskMapping:
    examples: list[XYExample] = []
    for x, y in zip(keys, values, strict=True):
        prompt = prompt_template.format(X=x)
        answer_ids = ensure_answer_token_length(tokenizer, y, max_answer_tokens)
        examples.append(
            XYExample(
                x=x,
                y=y,
                prompt=prompt,
                answer_token_ids=tuple(answer_ids),
            )
        )
    return TaskMapping(task_id=task_id, examples=tuple(examples))


def generate_task_pair(
    tokenizer: PreTrainedTokenizerBase,
    overlap: float,
    n_keys: int = 400,
    n_values: int = 400,
    seed: int = 0,
    prompt_template: str = "Q: {X}\nA:",
    max_answer_tokens: int = 2,
) -> TaskPair:
    """Create two tasks A/B with overlap and forced label conflicts.

    For shared keys, task-A and task-B always receive different values.
    """
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap must be in [0, 1].")
    if n_keys <= 0 or n_values <= 1:
        raise ValueError("n_keys must be > 0 and n_values must be > 1")
    if n_values < n_keys:
        raise ValueError("n_values must be >= n_keys to build disjoint A/B value pools.")

    rng = np.random.default_rng(seed)
    shared_n = int(round(overlap * n_keys))
    unique_n = n_keys - shared_n

    key_pool = select_clean_tokens(tokenizer, n=2 * n_keys, seed=seed + 11)
    val_pool = select_clean_tokens(tokenizer, n=2 * n_values, seed=seed + 23)

    shared_keys = list(key_pool[:shared_n])
    a_unique_keys = list(key_pool[shared_n : shared_n + unique_n])
    b_unique_keys = list(key_pool[n_keys : n_keys + unique_n])

    keys_a = shared_keys + a_unique_keys
    keys_b = shared_keys + b_unique_keys

    vals_a = list(val_pool[:n_keys])
    vals_b = list(val_pool[n_values : n_values + n_keys])

    rng.shuffle(vals_a)
    rng.shuffle(vals_b)

    # Resolve all shared-key conflicts: guarantee vals_a[i] != vals_b[i] for i < shared_n.
    max_iters = shared_n * n_keys  # safety bound
    changed = True
    iters = 0
    while changed and iters < max_iters:
        changed = False
        for idx in range(shared_n):
            if vals_a[idx] == vals_b[idx]:
                # Find a non-shared position to swap with that doesn't create a new conflict.
                swapped = False
                for candidate in range(shared_n, n_keys):
                    if vals_b[candidate] != vals_a[idx]:
                        vals_b[idx], vals_b[candidate] = vals_b[candidate], vals_b[idx]
                        swapped = True
                        break
                if not swapped:
                    # Fallback: swap with next position (original behaviour).
                    swap_idx = (idx + 1) % n_keys
                    vals_b[idx], vals_b[swap_idx] = vals_b[swap_idx], vals_b[idx]
                    changed = True
                iters += 1

    task_a = _build_task(
        task_id="A",
        keys=keys_a,
        values=vals_a,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_answer_tokens=max_answer_tokens,
    )
    task_b = _build_task(
        task_id="B",
        keys=keys_b,
        values=vals_b,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_answer_tokens=max_answer_tokens,
    )

    return TaskPair(
        task_a=task_a,
        task_b=task_b,
        shared_keys=tuple(shared_keys),
        conflict_keys=tuple(shared_keys),
    )
