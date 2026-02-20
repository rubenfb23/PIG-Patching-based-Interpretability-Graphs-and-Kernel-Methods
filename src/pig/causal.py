"""Interventional causal edge evaluation utilities.

This module implements an MVP causal evaluation stack with three levels:

- Level A: internal influence I(u->v) measured on destination activations
- Level B: restoration/mediation metrics on model outputs
- Level C: path blocking via clamping destination nodes to baseline cache
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from pig.graph import Node
from pig.model import (
    ALLOWED_NODE_TYPES,
    NODE_TYPE_ATT,
    ActivationCache,
    HookedModel,
)
from pig.patching import PatchEffectTensor
from pig.prompts import PromptPair


PatchNode = tuple[int, int, str, Optional[int]]


def _validate_node_type(node_type: str) -> None:
    if node_type not in ALLOWED_NODE_TYPES:
        raise ValueError(f"Unsupported node_type: {node_type}")


@dataclass(frozen=True)
class CausalEdgeCandidate:
    """Directed candidate edge for interventional causal evaluation."""

    src_layer: int
    src_token: int
    src_node_type: str = NODE_TYPE_ATT
    src_head: Optional[int] = None
    dst_layer: int = 0
    dst_token: int = 0
    dst_node_type: str = NODE_TYPE_ATT
    dst_head: Optional[int] = None
    correlation_score: float = 0.0
    rank: int = 0

    def __post_init__(self) -> None:
        _validate_node_type(self.src_node_type)
        _validate_node_type(self.dst_node_type)
        if self.src_node_type == NODE_TYPE_ATT and self.src_head is None:
            raise ValueError("src_head is required for attention nodes")
        if self.dst_node_type == NODE_TYPE_ATT and self.dst_head is None:
            raise ValueError("dst_head is required for attention nodes")

    def source_node(self) -> PatchNode:
        return (
            self.src_layer,
            self.src_token,
            self.src_node_type,
            self.src_head,
        )

    def target_node(self) -> PatchNode:
        return (
            self.dst_layer,
            self.dst_token,
            self.dst_node_type,
            self.dst_head,
        )

    def edge_id(self) -> str:
        return f"{_node_label(self.source_node())}->{_node_label(self.target_node())}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CausalEvalConfig:
    """Configuration for causal edge evaluation."""

    eps: float = 1e-6
    bootstrap_samples: int = 200
    permutation_samples: int = 200
    ci_alpha: float = 0.05
    seed: int = 42
    mediation_zero_tol: float = 0.05


@dataclass
class CausalEvalResult:
    """Structured output for a causal evaluation run."""

    model_name: str
    config: CausalEvalConfig
    run_metadata: dict
    edges: list[dict]
    arrays: dict[str, NDArray[np.float64] | NDArray[np.str_]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": self.model_name,
            "config": asdict(self.config),
            "run_metadata": self.run_metadata,
            "edges": self.edges,
        }

    def save_json(self, path: Path | str) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return output_path

    def save_npz(self, path: Path | str) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **self.arrays)
        return output_path


def restoration_fraction(
    patched_score: float,
    base_score: float,
    clean_score: float,
    eps: float = 1e-6,
) -> float:
    """R(S) = (O(intervened) - O(base)) / (O(clean) - O(base) + eps)."""
    denominator = (clean_score - base_score) + eps
    return float((patched_score - base_score) / denominator)


def mediation_score(r_uv: float, r_v: float) -> float:
    """M(u->v) = R({u,v}) - R({v})."""
    return float(r_uv - r_v)


def necessity_score(r_u: float, r_u_clamp_v: float) -> float:
    """necessity(u->v) = R({u}) - R({u, clamp(v=base)})."""
    return float(r_u - r_u_clamp_v)


def activation_influence_score(
    patched_v: NDArray[np.float32] | NDArray[np.float64],
    base_v: NDArray[np.float32] | NDArray[np.float64],
    clean_v: NDArray[np.float32] | NDArray[np.float64],
    eps: float = 1e-6,
) -> float:
    """I(u->v) based on normalized L2 displacement at destination node v."""
    numerator = float(np.linalg.norm(np.asarray(patched_v) - np.asarray(base_v)))
    denominator = float(np.linalg.norm(np.asarray(clean_v) - np.asarray(base_v))) + eps
    return numerator / denominator


def _node_label(node: PatchNode) -> str:
    layer, token, node_type, head = node
    if node_type == NODE_TYPE_ATT:
        return f"L{layer}T{token}.{node_type}[{head}]"
    return f"L{layer}T{token}.{node_type}"


def _get_activation(cache: ActivationCache, node: PatchNode):
    layer, token, node_type, head = node
    return cache.get(layer, token, node_type=node_type, head=head)


def _bootstrap_mean_ci(
    values: NDArray[np.float64],
    samples: int,
    alpha: float,
    seed: int,
) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if samples <= 0 or values.size == 1:
        return mean, mean, mean

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(samples, values.size))
    boot_means = values[draws].mean(axis=1)
    low = float(np.quantile(boot_means, alpha / 2.0))
    high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return mean, low, high


def _sign_flip_pvalue(values: NDArray[np.float64], samples: int, seed: int) -> float:
    if values.size == 0 or samples <= 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    observed = abs(float(np.mean(values)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, values.size))
    permuted = np.abs((values[None, :] * signs).mean(axis=1))
    pvalue = (np.sum(permuted >= observed) + 1.0) / (samples + 1.0)
    return float(pvalue)


def _metric_summary(
    values: list[float],
    config: CausalEvalConfig,
    seed: int,
) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    mean, ci_low, ci_high = _bootstrap_mean_ci(
        array,
        samples=config.bootstrap_samples,
        alpha=config.ci_alpha,
        seed=seed,
    )
    p_value = _sign_flip_pvalue(
        array,
        samples=config.permutation_samples,
        seed=seed + 1,
    )
    return {
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n": int(array.size),
        "values": array.tolist(),
    }


def load_cached_patch_effect_tensors(
    cache_dir: Path | str = ".cache/patch_effects",
) -> list[PatchEffectTensor]:
    """Load all cached patch-effect tensors from disk."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return []

    tensors: list[PatchEffectTensor] = []
    for json_path in sorted(cache_path.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tensors.append(PatchEffectTensor.from_dict(payload))
    return tensors


def select_causal_subset_from_cache(
    tensors: Sequence[PatchEffectTensor],
    num_examples: int = 20,
    component_size: int = 14,
    seed: int = 42,
    require_clean_better: bool = True,
) -> list[PatchEffectTensor]:
    """Deterministically sample a fast/reproducible subset for causal eval."""
    filtered = [
        t
        for t in tensors
        if len(t.component_axis) == component_size
        and (not require_clean_better or t.clean_score > t.base_score)
    ]
    filtered = sorted(
        filtered,
        key=lambda t: (
            str(t.prompt_pair.slice_label),
            t.prompt_pair.x_cln,
            t.prompt_pair.x_crp,
            t.prompt_pair.y_star,
        ),
    )

    if num_examples <= 0:
        return []
    if len(filtered) <= num_examples:
        return filtered

    rng = np.random.default_rng(seed)
    selected_indices = np.sort(
        rng.choice(len(filtered), size=num_examples, replace=False)
    )
    return [filtered[int(idx)] for idx in selected_indices]


def propose_causal_edge_candidates(
    tensors: Sequence[PatchEffectTensor],
    num_edges: int = 5,
    node_types: Sequence[str] = (NODE_TYPE_ATT,),
    enforce_direction: bool = True,
    eps: float = 1e-6,
) -> list[CausalEdgeCandidate]:
    """Propose directed candidate edges from normalized patch effects."""
    if not tensors:
        return []
    if num_edges <= 0:
        return []

    for node_type in node_types:
        _validate_node_type(node_type)

    component_axis = tensors[0].component_axis
    if not component_axis:
        return []

    num_layers = tensors[0].num_layers
    min_tokens = min(t.num_tokens for t in tensors)
    num_components = len(component_axis)

    normalized_rows: list[NDArray[np.float64]] = []
    for tensor in tensors:
        delta = max(float(tensor.clean_score - tensor.base_score), eps)
        normalized = tensor.effects[:, :min_tokens, :] / delta
        normalized_rows.append(normalized.reshape(-1).astype(np.float64, copy=False))

    matrix = np.stack(normalized_rows, axis=0)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    std = np.std(centered, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = centered / std
    corr = (normalized.T @ normalized) / max(1, normalized.shape[0])
    corr = corr.astype(np.float64, copy=False)
    np.fill_diagonal(corr, 0.0)

    nodes = [
        Node.from_index(i, num_tokens=min_tokens, component_axis=component_axis)
        for i in range(num_layers * min_tokens * num_components)
    ]
    allowed_types = set(node_types)
    valid_indices = [
        index
        for index, node in enumerate(nodes)
        if node.node_type in allowed_types
    ]

    scored: list[tuple[float, float, int, int]] = []
    for src_idx in valid_indices:
        src = nodes[src_idx]
        for dst_idx in valid_indices:
            if src_idx == dst_idx:
                continue
            dst = nodes[dst_idx]
            if enforce_direction and not (src < dst):
                continue
            weight = float(corr[src_idx, dst_idx])
            if not np.isfinite(weight):
                continue
            scored.append((abs(weight), weight, src_idx, dst_idx))

    scored.sort(key=lambda item: (-item[0], -abs(item[1]), item[2], item[3]))

    candidates: list[CausalEdgeCandidate] = []
    for rank, (_, weight, src_idx, dst_idx) in enumerate(scored[:num_edges], start=1):
        src = nodes[src_idx]
        dst = nodes[dst_idx]
        candidates.append(
            CausalEdgeCandidate(
                src_layer=src.layer,
                src_token=src.token,
                src_node_type=src.node_type,
                src_head=src.head,
                dst_layer=dst.layer,
                dst_token=dst.token,
                dst_node_type=dst.node_type,
                dst_head=dst.head,
                correlation_score=weight,
                rank=rank,
            )
        )

    return candidates


def evaluate_causal_edges(
    model: HookedModel,
    prompt_pairs: Sequence[PromptPair],
    candidates: Sequence[CausalEdgeCandidate],
    config: CausalEvalConfig,
    *,
    show_progress: bool = False,
    progress_desc: str = "causal-eval",
    progress_file = None,
) -> CausalEvalResult:
    """Run causal Level A/B/C edge evaluation with real interventions."""
    if not prompt_pairs:
        raise ValueError("prompt_pairs must not be empty")
    if not candidates:
        raise ValueError("candidates must not be empty")
    if not hasattr(model, "patched_score_multi_with_capture"):
        raise TypeError(
            "model must implement patched_score_multi_with_capture for causal eval"
        )

    required_node_types = sorted(
        {
            edge.src_node_type
            for edge in candidates
        }
        | {
            edge.dst_node_type
            for edge in candidates
        }
    )

    example_contexts = []
    for pair in prompt_pairs:
        clean_cache = model.cache_clean_components(pair.x_cln, node_types=required_node_types)
        base_cache = model.cache_clean_components(pair.x_crp, node_types=required_node_types)
        clean_score = float(model.score(pair.x_cln, pair.y_star))
        base_score = float(model.score(pair.x_crp, pair.y_star))
        example_contexts.append(
            {
                "pair": pair,
                "clean_cache": clean_cache,
                "base_cache": base_cache,
                "clean_score": clean_score,
                "base_score": base_score,
            }
        )

    edge_results: list[dict] = []
    edge_ids: list[str] = []
    src_labels: list[str] = []
    dst_labels: list[str] = []
    i_mean: list[float] = []
    i_low: list[float] = []
    i_high: list[float] = []
    r_u_mean: list[float] = []
    r_v_mean: list[float] = []
    r_uv_mean: list[float] = []
    m_mean: list[float] = []
    r_u_clamp_mean: list[float] = []
    necessity_mean_values: list[float] = []
    necessity_low: list[float] = []
    necessity_high: list[float] = []

    skipped_edges = 0

    progress = None
    if show_progress:
        progress = tqdm(
            total=len(candidates),
            desc=progress_desc,
            dynamic_ncols=True,
            file=progress_file if progress_file is not None else sys.stderr,
        )

    try:
        for edge_index, edge in enumerate(candidates):
            src_node = edge.source_node()
            dst_node = edge.target_node()

            i_values: list[float] = []
            r_u_values: list[float] = []
            r_v_values: list[float] = []
            r_uv_values: list[float] = []
            m_values: list[float] = []
            r_u_clamp_values: list[float] = []
            necessity_values: list[float] = []

            for context in example_contexts:
                pair = context["pair"]
                clean_cache = context["clean_cache"]
                base_cache = context["base_cache"]
                clean_score = context["clean_score"]
                base_score = context["base_score"]

                base_v = _get_activation(base_cache, dst_node)
                clean_v = _get_activation(clean_cache, dst_node)
                if base_v is None or clean_v is None:
                    continue

                score_u, patched_capture = model.patched_score_multi_with_capture(
                    pair.x_crp,
                    pair.y_star,
                    patch_cache=clean_cache,
                    patch_nodes={src_node},
                    capture_nodes={dst_node},
                )
                patched_v = _get_activation(patched_capture, dst_node)
                if patched_v is None:
                    continue

                i_val = activation_influence_score(
                    patched_v.cpu().numpy(),
                    base_v.cpu().numpy(),
                    clean_v.cpu().numpy(),
                    eps=config.eps,
                )
                r_u = restoration_fraction(
                    score_u,
                    base_score=base_score,
                    clean_score=clean_score,
                    eps=config.eps,
                )

                score_v = model.patched_score_multi(
                    pair.x_crp,
                    pair.y_star,
                    clean_cache,
                    {dst_node},
                )
                r_v = restoration_fraction(
                    score_v,
                    base_score=base_score,
                    clean_score=clean_score,
                    eps=config.eps,
                )

                score_uv = model.patched_score_multi(
                    pair.x_crp,
                    pair.y_star,
                    clean_cache,
                    {src_node, dst_node},
                )
                r_uv = restoration_fraction(
                    score_uv,
                    base_score=base_score,
                    clean_score=clean_score,
                    eps=config.eps,
                )

                score_u_clamp, _ = model.patched_score_multi_with_capture(
                    pair.x_crp,
                    pair.y_star,
                    patch_cache=clean_cache,
                    patch_nodes={src_node},
                    clamp_cache=base_cache,
                    clamp_nodes={dst_node},
                )
                r_u_clamp = restoration_fraction(
                    score_u_clamp,
                    base_score=base_score,
                    clean_score=clean_score,
                    eps=config.eps,
                )

                i_values.append(i_val)
                r_u_values.append(r_u)
                r_v_values.append(r_v)
                r_uv_values.append(r_uv)
                m_values.append(mediation_score(r_uv, r_v))
                r_u_clamp_values.append(r_u_clamp)
                necessity_values.append(necessity_score(r_u, r_u_clamp))

            if not i_values:
                skipped_edges += 1
                if progress is not None:
                    progress.update(1)
                continue

            base_seed = config.seed + edge_index * 101
            i_summary = _metric_summary(i_values, config=config, seed=base_seed + 1)
            r_u_summary = _metric_summary(r_u_values, config=config, seed=base_seed + 2)
            r_v_summary = _metric_summary(r_v_values, config=config, seed=base_seed + 3)
            r_uv_summary = _metric_summary(r_uv_values, config=config, seed=base_seed + 4)
            m_summary = _metric_summary(m_values, config=config, seed=base_seed + 5)
            r_u_clamp_summary = _metric_summary(
                r_u_clamp_values,
                config=config,
                seed=base_seed + 6,
            )
            necessity_summary = _metric_summary(
                necessity_values,
                config=config,
                seed=base_seed + 7,
            )

            classification = (
                "mediated"
                if abs(m_summary["mean"]) <= config.mediation_zero_tol
                else "parallel_or_synergy"
            )
            coherence_drop = r_u_clamp_summary["mean"] < r_u_summary["mean"]

            edge_entry = {
                "edge_id": edge.edge_id(),
                "candidate": edge.to_dict(),
                "level_a": {"I": i_summary},
                "level_b": {
                    "R_u": r_u_summary,
                    "R_v": r_v_summary,
                    "R_uv": r_uv_summary,
                    "M": m_summary,
                },
                "level_c": {
                    "R_u_clamp_v": r_u_clamp_summary,
                    "necessity": necessity_summary,
                },
                "diagnostics": {
                    "classification": classification,
                    "coherence_drop_with_clamp": bool(coherence_drop),
                },
            }
            edge_results.append(edge_entry)

            edge_ids.append(edge.edge_id())
            src_labels.append(_node_label(src_node))
            dst_labels.append(_node_label(dst_node))
            i_mean.append(i_summary["mean"])
            i_low.append(i_summary["ci_low"])
            i_high.append(i_summary["ci_high"])
            r_u_mean.append(r_u_summary["mean"])
            r_v_mean.append(r_v_summary["mean"])
            r_uv_mean.append(r_uv_summary["mean"])
            m_mean.append(m_summary["mean"])
            r_u_clamp_mean.append(r_u_clamp_summary["mean"])
            necessity_mean_values.append(necessity_summary["mean"])
            necessity_low.append(necessity_summary["ci_low"])
            necessity_high.append(necessity_summary["ci_high"])
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    if not edge_results:
        raise RuntimeError("No valid causal edges were evaluated")

    arrays: dict[str, NDArray[np.float64] | NDArray[np.str_]] = {
        "edge_ids": np.asarray(edge_ids, dtype=np.str_),
        "src_labels": np.asarray(src_labels, dtype=np.str_),
        "dst_labels": np.asarray(dst_labels, dtype=np.str_),
        "I_mean": np.asarray(i_mean, dtype=np.float64),
        "I_ci_low": np.asarray(i_low, dtype=np.float64),
        "I_ci_high": np.asarray(i_high, dtype=np.float64),
        "R_u_mean": np.asarray(r_u_mean, dtype=np.float64),
        "R_v_mean": np.asarray(r_v_mean, dtype=np.float64),
        "R_uv_mean": np.asarray(r_uv_mean, dtype=np.float64),
        "M_mean": np.asarray(m_mean, dtype=np.float64),
        "R_u_clamp_v_mean": np.asarray(r_u_clamp_mean, dtype=np.float64),
        "necessity_mean": np.asarray(necessity_mean_values, dtype=np.float64),
        "necessity_ci_low": np.asarray(necessity_low, dtype=np.float64),
        "necessity_ci_high": np.asarray(necessity_high, dtype=np.float64),
    }

    metadata = {
        "num_prompt_pairs_requested": len(prompt_pairs),
        "num_prompt_pairs_used": len(example_contexts),
        "num_candidates_requested": len(candidates),
        "num_edges_evaluated": len(edge_results),
        "num_edges_skipped": skipped_edges,
        "required_node_types": required_node_types,
    }

    return CausalEvalResult(
        model_name=getattr(model, "model_name", "unknown"),
        config=config,
        run_metadata=metadata,
        edges=edge_results,
        arrays=arrays,
    )
