"""Layer-wise causal subspace extraction and projector caching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from clmi.data.synth_tasks import TaskMapping
from clmi.utils.config import SubspaceConfig
from clmi.utils.io import ensure_dir, stable_hash


@dataclass
class ProjectorPack:
    """Bundle of orthonormal bases and projectors by layer."""

    task_id: str
    layers: tuple[int, ...]
    bases: dict[int, np.ndarray]
    projectors: dict[int, np.ndarray]
    logit_diff: np.ndarray


def fit_layer_subspace(
    h_matrix: np.ndarray,
    y: np.ndarray,
    k: int,
    ridge_alpha: float,
    bootstrap_iters: int,
    seed: int,
    layer_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one layer subspace using bootstrap ridge vectors + QR."""
    n, d_model = h_matrix.shape
    if n < 2:
        raise ValueError("Need at least two samples to fit subspace.")

    rng = np.random.default_rng(seed + layer_idx)
    iters = max(bootstrap_iters, k)
    vectors: list[np.ndarray] = []

    for _ in range(iters):
        idx = rng.choice(n, size=n, replace=True)
        x_boot = h_matrix[idx]
        y_boot = y[idx]

        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(x_boot, y_boot)
        w = ridge.coef_.astype(np.float64)

        norm = np.linalg.norm(w)
        if norm > 1e-12:
            w = w / norm
        vectors.append(w)

    W = np.stack(vectors, axis=1)
    k_eff = min(k, d_model)
    Q, _ = np.linalg.qr(W)
    U = Q[:, :k_eff]
    P = U @ U.T
    return U.astype(np.float32), P.astype(np.float32)


@torch.no_grad()
def collect_layer_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task: TaskMapping,
    layers: list[int],
    device: str,
    negative_token_ids: list[int] | None = None,
    batch_size: int = 32,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Collect h_{l,i} and y_i=logit_diff for a task.

    h_{l,i} is taken at the final prompt position (before first answer token).
    Uses batched inference for efficiency.
    """
    model.eval()
    activations: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    y_values: list[float] = []

    first_answer_ids = [int(ex.answer_token_ids[0]) for ex in task.examples]
    if negative_token_ids is None:
        rotated = first_answer_ids[1:] + first_answer_ids[:1]
        negative_token_ids = [
            neg if neg != pos else (neg + 1) % model.config.vocab_size
            for pos, neg in zip(first_answer_ids, rotated, strict=True)
        ]

    examples = list(task.examples)
    n = len(examples)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_examples = examples[start:end]
        batch_pos_toks = first_answer_ids[start:end]
        batch_neg_toks = negative_token_ids[start:end]

        # Build padded batch of full sequences (prompt + answer)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        all_ids: list[list[int]] = []
        prompt_lengths: list[int] = []
        for ex in batch_examples:
            prompt_ids = tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
            answer_ids = list(ex.answer_token_ids)
            all_ids.append(prompt_ids + answer_ids)
            prompt_lengths.append(len(prompt_ids))

        max_len = max(len(ids) for ids in all_ids)
        input_tensor = torch.full(
            (len(all_ids), max_len), pad_id, dtype=torch.long, device=device
        )
        for i, ids in enumerate(all_ids):
            input_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        attention_mask = (input_tensor != pad_id).long()
        outputs = model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        logits = outputs.logits  # [B, T, V]
        for i in range(len(batch_examples)):
            pos = prompt_lengths[i] - 1
            y = float(
                logits[i, pos, batch_pos_toks[i]].item()
                - logits[i, pos, batch_neg_toks[i]].item()
            )
            y_values.append(y)

            for layer in layers:
                hidden = outputs.hidden_states[layer + 1][i, pos, :].detach().cpu().numpy()
                activations[layer].append(hidden)

    stacked = {layer: np.stack(values, axis=0) for layer, values in activations.items()}
    return stacked, np.asarray(y_values, dtype=np.float32)


def _cache_paths(cache_root: Path, cache_key: str) -> tuple[Path, Path]:
    cache_dir = ensure_dir(cache_root)
    return cache_dir / f"{cache_key}.npz", cache_dir / f"{cache_key}.json"


def _activation_cache_path(cache_root: Path, cache_key: str) -> Path:
    return ensure_dir(cache_root) / f"{cache_key}_activations.npz"


def _make_cache_key(
    task: TaskMapping,
    cfg: SubspaceConfig,
    model_name: str,
    state_tag: str,
) -> str:
    alpha_repr = ",".join(f"{a:.6f}" for a in cfg.alpha_layers)
    layers_repr = ",".join(str(l) for l in cfg.layers)
    parts = [
        model_name,
        state_tag,
        task.task_id,
        str(len(task.examples)),
        layers_repr,
        str(cfg.k),
        str(cfg.ridge_alpha),
        str(cfg.bootstrap_iters),
        alpha_repr,
    ]
    return stable_hash(parts)


def load_projector_pack(
    cache_root: Path,
    cache_key: str,
) -> ProjectorPack | None:
    """Load cached projector pack if present."""
    npz_path, meta_path = _cache_paths(cache_root, cache_key)
    if not npz_path.exists() or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    data = np.load(npz_path)

    layers = tuple(meta["layers"])
    bases = {layer: data[f"U_{layer}"] for layer in layers}
    projectors = {layer: data[f"P_{layer}"] for layer in layers}
    y = data["y"]

    return ProjectorPack(
        task_id=meta["task_id"],
        layers=layers,
        bases=bases,
        projectors=projectors,
        logit_diff=y,
    )


def save_projector_pack(
    cache_root: Path,
    cache_key: str,
    pack: ProjectorPack,
) -> None:
    """Persist projector pack to cache."""
    npz_path, meta_path = _cache_paths(cache_root, cache_key)

    payload: dict[str, np.ndarray] = {"y": pack.logit_diff}
    for layer in pack.layers:
        payload[f"U_{layer}"] = pack.bases[layer]
        payload[f"P_{layer}"] = pack.projectors[layer]

    np.savez_compressed(npz_path, **payload)
    meta_path.write_text(
        json.dumps(
            {
                "task_id": pack.task_id,
                "layers": list(pack.layers),
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_task_projectors(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task: TaskMapping,
    cfg: SubspaceConfig,
    device: str,
    cache_root: Path,
    model_name: str,
    state_tag: str,
) -> tuple[ProjectorPack, bool]:
    """Build/load projector pack for a task.

    Returns:
        (projector_pack, cache_hit)
    """
    layers = list(cfg.layers)
    if not layers:
        layers = list(range(model.config.n_layer))

    cache_key = _make_cache_key(task, cfg, model_name=model_name, state_tag=state_tag)
    cached = load_projector_pack(cache_root, cache_key)
    if cached is not None:
        return cached, True

    acts_cache = _activation_cache_path(cache_root, cache_key)
    if acts_cache.exists():
        loaded = np.load(acts_cache)
        activations = {layer: loaded[f"H_{layer}"] for layer in layers}
        y = loaded["y"]
    else:
        activations, y = collect_layer_activations(
            model,
            tokenizer,
            task,
            layers=layers,
            device=device,
        )
        payload = {"y": y}
        payload.update({f"H_{layer}": activations[layer] for layer in layers})
        np.savez_compressed(acts_cache, **payload)

    bases: dict[int, np.ndarray] = {}
    projectors: dict[int, np.ndarray] = {}

    base_seed = hash((model_name, state_tag, task.task_id)) & 0xFFFFFFFF
    for i, layer in enumerate(layers):
        U, P = fit_layer_subspace(
            activations[layer],
            y,
            k=cfg.k,
            ridge_alpha=cfg.ridge_alpha,
            bootstrap_iters=cfg.bootstrap_iters,
            seed=base_seed,
            layer_idx=i,
        )
        bases[layer] = U
        projectors[layer] = P

    pack = ProjectorPack(
        task_id=task.task_id,
        layers=tuple(layers),
        bases=bases,
        projectors=projectors,
        logit_diff=y,
    )
    save_projector_pack(cache_root, cache_key, pack)
    return pack, False


def sample_random_projectors(
    layers: list[int],
    d_model: int,
    k: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Sample random orthonormal projector controls with same rank k."""
    rng = np.random.default_rng(seed)
    rank = min(k, d_model)
    projectors: dict[int, np.ndarray] = {}
    for layer in layers:
        random_matrix = rng.normal(size=(d_model, rank))
        Q, _ = np.linalg.qr(random_matrix)
        U = Q[:, :rank]
        projectors[layer] = (U @ U.T).astype(np.float32)
    return projectors
