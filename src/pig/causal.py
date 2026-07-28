"""Interventional causal edge evaluation utilities.

This module implements an MVP causal evaluation stack with three levels:

- Level A: internal influence I(u->v) measured on destination activations
- Level B: restoration/mediation metrics on model outputs
- Level C: path blocking via clamping destination nodes to baseline cache
"""

from __future__ import annotations

from collections import Counter, defaultdict
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
    NODE_TYPE_RES,
    ActivationCache,
    HookedModel,
)
from pig.patching import PatchEffectTensor, build_axis_fingerprint
from pig.prompts import PromptPair, get_prompt_pair_distractor


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


def _fdr_bh_adjust(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjusted p-values."""
    if not p_values:
        return []
    raw = np.asarray(p_values, dtype=np.float64)
    m = raw.size
    order = np.argsort(raw)
    sorted_p = raw[order]
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        candidate = float(sorted_p[i]) * m / rank
        running_min = min(running_min, candidate)
        adjusted_sorted[i] = running_min
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    return adjusted.tolist()


def _bonferroni_adjust(p_values: Sequence[float]) -> list[float]:
    """Bonferroni adjusted p-values."""
    if not p_values:
        return []
    m = len(p_values)
    adjusted = [min(float(p) * m, 1.0) for p in p_values]
    return adjusted


def _apply_multiple_testing_corrections(
    edge_results: list[dict],
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, int | float]]:
    """Attach FDR/BH and Bonferroni corrections for edge-wise metric p-values."""
    metric_paths = {
        "I": ("level_a", "I"),
        "R_u": ("level_b", "R_u"),
        "R_v": ("level_b", "R_v"),
        "R_uv": ("level_b", "R_uv"),
        "M": ("level_b", "M"),
        "R_u_clamp_v": ("level_c", "R_u_clamp_v"),
        "necessity": ("level_c", "necessity"),
    }
    correction_summary: dict[str, dict[str, int | float]] = {}
    for metric_name, (section, metric_key) in metric_paths.items():
        valid_indices: list[int] = []
        raw_p_values: list[float] = []
        for idx, edge in enumerate(edge_results):
            summary = edge.get(section, {}).get(metric_key, {})
            p_value = summary.get("p_value")
            if p_value is None:
                continue
            p_value_float = float(p_value)
            if not np.isfinite(p_value_float):
                continue
            valid_indices.append(idx)
            raw_p_values.append(p_value_float)

        fdr_adjusted = _fdr_bh_adjust(raw_p_values)
        bonf_adjusted = _bonferroni_adjust(raw_p_values)
        for edge_idx, p_fdr, p_bonf in zip(
            valid_indices, fdr_adjusted, bonf_adjusted
        ):
            metric_summary = edge_results[edge_idx][section][metric_key]
            metric_summary["p_value_fdr_bh"] = float(p_fdr)
            metric_summary["p_value_bonferroni"] = float(p_bonf)
            metric_summary["significant_fdr_bh"] = bool(p_fdr <= alpha)
            metric_summary["significant_bonferroni"] = bool(p_bonf <= alpha)

        correction_summary[metric_name] = {
            "raw_tests": len(raw_p_values),
            "alpha": float(alpha),
            "significant_fdr_bh": int(sum(p <= alpha for p in fdr_adjusted)),
            "significant_bonferroni": int(sum(p <= alpha for p in bonf_adjusted)),
        }
    return correction_summary


def summarize_clean_base_effects_by_slice(
    tensors: Sequence[PatchEffectTensor],
    *,
    bootstrap_samples: int = 200,
    ci_alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Summarize clean-base effect deltas overall and by slice."""
    deltas = [float(t.clean_score - t.base_score) for t in tensors]
    all_values = np.asarray(deltas, dtype=np.float64)
    all_values = all_values[np.isfinite(all_values)]
    mean, ci_low, ci_high = _bootstrap_mean_ci(
        all_values,
        samples=bootstrap_samples,
        alpha=ci_alpha,
        seed=seed,
    )
    summary: dict[str, object] = {
        "n": int(all_values.size),
        "mean_clean_minus_base": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "by_slice": {},
    }
    grouped: dict[str, list[float]] = defaultdict(list)
    for tensor in tensors:
        slice_id = _slice_stratum_id(tensor)
        grouped[slice_id].append(float(tensor.clean_score - tensor.base_score))

    by_slice: dict[str, dict[str, float | int]] = {}
    for idx, slice_id in enumerate(sorted(grouped)):
        values = np.asarray(grouped[slice_id], dtype=np.float64)
        values = values[np.isfinite(values)]
        slice_mean, slice_low, slice_high = _bootstrap_mean_ci(
            values,
            samples=bootstrap_samples,
            alpha=ci_alpha,
            seed=seed + idx + 1,
        )
        by_slice[slice_id] = {
            "n": int(values.size),
            "mean_clean_minus_base": slice_mean,
            "ci_low": slice_low,
            "ci_high": slice_high,
        }
    summary["by_slice"] = by_slice
    return summary


def _slice_count_summary(
    tensors: Sequence[PatchEffectTensor],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for tensor in tensors:
        counts[_slice_stratum_id(tensor)] += 1
    return dict(sorted(counts.items()))


def _distribution_balance_metrics(
    counts: dict[str, int],
) -> dict[str, float | int | list[str] | None]:
    values = [int(count) for count in counts.values()]
    total = int(sum(values))
    if not values:
        return {
            "num_slices": 0,
            "total": 0,
            "max_count": 0,
            "min_count": 0,
            "max_share": 0.0,
            "min_share": 0.0,
            "max_to_min_ratio": None,
            "zero_count_slices": [],
        }

    max_count = max(values)
    min_count = min(values)
    zero_count_slices = [slice_id for slice_id, count in counts.items() if count == 0]
    nonzero = [count for count in values if count > 0]
    max_to_min_ratio: float | None
    if not nonzero:
        max_to_min_ratio = None
    else:
        max_to_min_ratio = float(max(nonzero) / min(nonzero))
    return {
        "num_slices": len(values),
        "total": total,
        "max_count": max_count,
        "min_count": min_count,
        "max_share": float(max_count / total) if total else 0.0,
        "min_share": float(min_count / total) if total else 0.0,
        "max_to_min_ratio": max_to_min_ratio,
        "zero_count_slices": zero_count_slices,
    }


def build_clean_better_filter_report(
    tensors: Sequence[PatchEffectTensor],
    *,
    bootstrap_samples: int = 200,
    ci_alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Report clean>base filter sensitivity with and without the filter."""
    all_tensors = list(tensors)
    filtered_tensors = [t for t in all_tensors if t.clean_score > t.base_score]
    all_counts = _slice_count_summary(all_tensors)
    filtered_counts = _slice_count_summary(filtered_tensors)
    slice_retention: dict[str, dict[str, float | int]] = {}
    total_all = len(all_tensors)
    total_filtered = len(filtered_tensors)
    for slice_id in sorted(set(all_counts) | set(filtered_counts)):
        input_n = int(all_counts.get(slice_id, 0))
        retained_n = int(filtered_counts.get(slice_id, 0))
        discarded_n = input_n - retained_n
        share_before = float(input_n / total_all) if total_all else 0.0
        share_after = float(retained_n / total_filtered) if total_filtered else 0.0
        slice_retention[slice_id] = {
            "input_n": input_n,
            "retained_n": retained_n,
            "discarded_n": discarded_n,
            "retention_rate": float(retained_n / input_n) if input_n else 0.0,
            "share_before": share_before,
            "share_after": share_after,
            "share_shift": share_after - share_before,
        }
    return {
        "rule": "clean_score > base_score",
        "input_tensors": len(all_tensors),
        "retained_with_filter": len(filtered_tensors),
        "discarded_by_filter": len(all_tensors) - len(filtered_tensors),
        "retention_rate": (
            float(len(filtered_tensors) / len(all_tensors)) if all_tensors else 0.0
        ),
        "without_filter": summarize_clean_base_effects_by_slice(
            all_tensors,
            bootstrap_samples=bootstrap_samples,
            ci_alpha=ci_alpha,
            seed=seed,
        ),
        "with_filter": summarize_clean_base_effects_by_slice(
            filtered_tensors,
            bootstrap_samples=bootstrap_samples,
            ci_alpha=ci_alpha,
            seed=seed + 1000,
        ),
        "slice_retention": slice_retention,
        "balance": {
            "before_filter": _distribution_balance_metrics(
                {
                    slice_id: metrics["input_n"]
                    for slice_id, metrics in slice_retention.items()
                }
            ),
            "after_filter": _distribution_balance_metrics(
                {
                    slice_id: metrics["retained_n"]
                    for slice_id, metrics in slice_retention.items()
                }
            ),
        },
    }


def load_cached_patch_effect_tensors(
    cache_dir: Path | str = ".cache/patch_effects",
    *,
    expected_model_name: str | None = None,
    expected_model_fingerprint: str | None = None,
    expected_axis_fingerprint: str | None = None,
    allow_legacy_cache: bool = False,
    component_size: int | None = None,
    return_stats: bool = False,
) -> list[PatchEffectTensor] | tuple[list[PatchEffectTensor], dict]:
    """Load cached patch-effect tensors with strict compatibility filtering."""
    cache_path = Path(cache_dir)
    reason_counts: Counter[str] = Counter()
    stats: dict[str, object] = {
        "cache_dir": str(cache_path),
        "total_files": 0,
        "loaded_tensors": 0,
        "accepted_tensors": 0,
        "accepted_legacy_tensors": 0,
        "discarded_tensors": 0,
        "allow_legacy_cache": bool(allow_legacy_cache),
        "expected_model_name": expected_model_name,
        "expected_model_fingerprint": expected_model_fingerprint,
        "expected_axis_fingerprint": expected_axis_fingerprint,
        "component_size_validation": component_size,
    }
    if not cache_path.exists():
        stats["discard_reasons"] = {}
        if return_stats:
            return [], stats
        return []

    tensors: list[PatchEffectTensor] = []
    for json_path in sorted(cache_path.glob("*.json")):
        stats["total_files"] = int(stats["total_files"]) + 1
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            reason_counts["invalid_json"] += 1
            continue

        try:
            tensor = PatchEffectTensor.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            reason_counts["invalid_tensor_payload"] += 1
            continue
        stats["loaded_tensors"] = int(stats["loaded_tensors"]) + 1

        if tensor.is_legacy_cache_entry and not allow_legacy_cache:
            reason_counts["legacy_disallowed"] += 1
            continue

        if expected_model_name and tensor.model_name and tensor.model_name != expected_model_name:
            reason_counts["model_name_mismatch"] += 1
            continue

        if expected_model_fingerprint and not tensor.is_legacy_cache_entry:
            if tensor.model_fingerprint != expected_model_fingerprint:
                reason_counts["model_fingerprint_mismatch"] += 1
                continue

        candidate_axis_fingerprint = (
            tensor.axis_fingerprint or build_axis_fingerprint(tensor.component_axis)
        )
        if expected_axis_fingerprint and candidate_axis_fingerprint != expected_axis_fingerprint:
            reason_counts["axis_fingerprint_mismatch"] += 1
            continue

        if component_size is not None and len(tensor.component_axis) != component_size:
            reason_counts["component_size_validation_mismatch"] += 1
            continue

        tensors.append(tensor)
        if tensor.is_legacy_cache_entry:
            stats["accepted_legacy_tensors"] = int(stats["accepted_legacy_tensors"]) + 1

    stats["accepted_tensors"] = len(tensors)
    stats["discarded_tensors"] = sum(reason_counts.values())
    stats["discard_reasons"] = dict(sorted(reason_counts.items()))
    if return_stats:
        return tensors, stats
    return tensors


def select_causal_subset_from_cache(
    tensors: Sequence[PatchEffectTensor],
    num_examples: int = 20,
    component_size: int | None = None,
    seed: int = 42,
    require_clean_better: bool = True,
    return_stats: bool = False,
) -> list[PatchEffectTensor] | tuple[list[PatchEffectTensor], dict]:
    """Deterministically sample a fast/reproducible subset for causal eval.

    `component_size` is retained only as a deprecated validation check.
    """
    stats: dict[str, int | bool | None] = {
        "input_tensors": len(tensors),
        "require_clean_better": bool(require_clean_better),
        "component_size_validation": component_size,
        "discarded_clean_not_better": 0,
        "discarded_component_size_validation": 0,
        "candidates_after_filters": 0,
        "selected_count": 0,
    }
    filtered: list[PatchEffectTensor] = []
    for tensor in tensors:
        if require_clean_better and not (tensor.clean_score > tensor.base_score):
            stats["discarded_clean_not_better"] = int(
                stats["discarded_clean_not_better"]
            ) + 1
            continue
        if component_size is not None and len(tensor.component_axis) != component_size:
            stats["discarded_component_size_validation"] = int(
                stats["discarded_component_size_validation"]
            ) + 1
            continue
        filtered.append(tensor)
    stats["candidates_after_filters"] = len(filtered)
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
        if return_stats:
            return [], stats
        return []
    if len(filtered) <= num_examples:
        stats["selected_count"] = len(filtered)
        if return_stats:
            return filtered, stats
        return filtered

    rng = np.random.default_rng(seed)
    selected_indices = np.sort(
        rng.choice(len(filtered), size=num_examples, replace=False)
    )
    selected = [filtered[int(idx)] for idx in selected_indices]
    stats["selected_count"] = len(selected)
    if return_stats:
        return selected, stats
    return selected


def _slice_stratum_id(tensor: PatchEffectTensor) -> str:
    label = tensor.prompt_pair.slice_label
    return f"{label.task}:{label.corruption}"


def split_discovery_evaluation_tensors(
    tensors: Sequence[PatchEffectTensor],
    *,
    max_total_examples: int,
    seed: int = 42,
    discovery_fraction: float = 0.5,
    return_stats: bool = False,
) -> (
    tuple[list[PatchEffectTensor], list[PatchEffectTensor]]
    | tuple[list[PatchEffectTensor], list[PatchEffectTensor], dict]
):
    """Create disjoint discovery/evaluation subsets with slice-stratified pools."""
    stats: dict[str, object] = {
        "input_tensors": len(tensors),
        "max_total_examples": int(max_total_examples),
        "discovery_fraction": float(discovery_fraction),
        "seed": int(seed),
        "stratified_by": "slice_label.task+corruption",
        "strata_counts": {},
        "discovery_pool_count": 0,
        "evaluation_pool_count": 0,
        "selected_discovery_count": 0,
        "selected_evaluation_count": 0,
    }
    if max_total_examples <= 1 or len(tensors) <= 1:
        stats["error"] = (
            "Need at least 2 eligible tensors and max_total_examples > 1 "
            "for disjoint discovery/evaluation split."
        )
        if return_stats:
            return [], [], stats
        return [], []

    bounded_fraction = float(np.clip(discovery_fraction, 0.1, 0.9))
    sorted_tensors = sorted(
        tensors,
        key=lambda t: (
            str(t.prompt_pair.slice_label),
            t.prompt_pair.x_cln,
            t.prompt_pair.x_crp,
            t.prompt_pair.y_star,
        ),
    )
    strata: dict[str, list[PatchEffectTensor]] = defaultdict(list)
    for tensor in sorted_tensors:
        strata[_slice_stratum_id(tensor)].append(tensor)

    rng = np.random.default_rng(seed)
    discovery_pool: list[PatchEffectTensor] = []
    evaluation_pool: list[PatchEffectTensor] = []
    strata_counts: dict[str, dict[str, int]] = {}
    for stratum_id in sorted(strata):
        group = strata[stratum_id]
        permuted = list(rng.permutation(len(group)))
        shuffled_group = [group[i] for i in permuted]
        if len(group) == 1:
            # Keep singleton strata balanced between pools when possible.
            send_to_discovery = len(discovery_pool) <= len(evaluation_pool)
            if send_to_discovery:
                discovery_pool.append(shuffled_group[0])
                disc_count = 1
                eval_count = 0
            else:
                evaluation_pool.append(shuffled_group[0])
                disc_count = 0
                eval_count = 1
        else:
            disc_count = int(round(len(group) * bounded_fraction))
            disc_count = max(1, min(disc_count, len(group) - 1))
            eval_count = len(group) - disc_count
            discovery_pool.extend(shuffled_group[:disc_count])
            evaluation_pool.extend(shuffled_group[disc_count:])
        strata_counts[stratum_id] = {
            "total": len(group),
            "discovery_pool": disc_count,
            "evaluation_pool": eval_count,
        }

    if not discovery_pool and len(evaluation_pool) > 1:
        discovery_pool.append(evaluation_pool.pop())
    if not evaluation_pool and len(discovery_pool) > 1:
        evaluation_pool.append(discovery_pool.pop())

    stats["strata_counts"] = strata_counts
    stats["discovery_pool_count"] = len(discovery_pool)
    stats["evaluation_pool_count"] = len(evaluation_pool)
    if not discovery_pool or not evaluation_pool:
        stats["error"] = (
            "Unable to create non-empty disjoint discovery/evaluation pools. "
            "Increase cached examples across slices/corruptions."
        )
        if return_stats:
            return [], [], stats
        return [], []

    target_total = min(int(max_total_examples), len(sorted_tensors))
    target_discovery = max(1, int(round(target_total * bounded_fraction)))
    target_discovery = min(target_discovery, len(discovery_pool), target_total - 1)
    target_evaluation = target_total - target_discovery
    if target_evaluation <= 0:
        target_evaluation = 1
        target_discovery = min(len(discovery_pool), target_total - 1)

    if target_evaluation > len(evaluation_pool):
        deficit = target_evaluation - len(evaluation_pool)
        target_evaluation = len(evaluation_pool)
        target_discovery = min(len(discovery_pool), target_discovery + deficit)

    if target_discovery > len(discovery_pool):
        deficit = target_discovery - len(discovery_pool)
        target_discovery = len(discovery_pool)
        target_evaluation = min(len(evaluation_pool), target_evaluation + deficit)

    while target_discovery + target_evaluation < target_total:
        if len(discovery_pool) - target_discovery >= len(evaluation_pool) - target_evaluation:
            if target_discovery < len(discovery_pool):
                target_discovery += 1
                continue
        if target_evaluation < len(evaluation_pool):
            target_evaluation += 1
            continue
        break

    if target_discovery <= 0 or target_evaluation <= 0:
        stats["error"] = (
            "Unable to sample non-empty disjoint discovery/evaluation subsets. "
            "Increase --num-examples and cached examples."
        )
        if return_stats:
            return [], [], stats
        return [], []

    discovery_indices = np.sort(
        rng.choice(len(discovery_pool), size=target_discovery, replace=False)
    )
    evaluation_indices = np.sort(
        rng.choice(len(evaluation_pool), size=target_evaluation, replace=False)
    )
    discovery_subset = [discovery_pool[int(idx)] for idx in discovery_indices]
    evaluation_subset = [evaluation_pool[int(idx)] for idx in evaluation_indices]
    stats["selected_discovery_count"] = len(discovery_subset)
    stats["selected_evaluation_count"] = len(evaluation_subset)
    if return_stats:
        return discovery_subset, evaluation_subset, stats
    return discovery_subset, evaluation_subset


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
            # Attention heads at the same layer all read from the same residual
            # stream input and compute in parallel — patching one head cannot
            # influence another head in the same layer, so skip these edges.
            if (
                src.node_type == NODE_TYPE_ATT
                and dst.node_type == NODE_TYPE_ATT
                and src.layer == dst.layer
            ):
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


def propose_random_edge_candidates(
    tensors: Sequence[PatchEffectTensor],
    num_edges: int,
    node_types: Sequence[str] = (NODE_TYPE_ATT,),
    enforce_direction: bool = True,
    seed: int = 42,
) -> list[CausalEdgeCandidate]:
    """Propose random edge candidates for causal baseline comparison.

    Uses the same node set as ``propose_causal_edge_candidates`` but
    selects edges uniformly at random from valid edge slots. This
    provides a null baseline for the causal evaluation.
    """
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

    # Collect all valid edges
    valid_edges: list[tuple[int, int]] = []
    for src_idx in valid_indices:
        src = nodes[src_idx]
        for dst_idx in valid_indices:
            if src_idx == dst_idx:
                continue
            dst = nodes[dst_idx]
            if enforce_direction and not (src < dst):
                continue
            if (
                src.node_type == NODE_TYPE_ATT
                and dst.node_type == NODE_TYPE_ATT
                and src.layer == dst.layer
            ):
                continue
            valid_edges.append((src_idx, dst_idx))

    if not valid_edges:
        return []

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(valid_edges), size=min(num_edges, len(valid_edges)), replace=False)
    rng = np.random.default_rng(seed + 1)  # Shuffle chosen edges
    chosen = rng.permutation(chosen)

    candidates: list[CausalEdgeCandidate] = []
    for rank, idx in enumerate(chosen[:num_edges], start=1):
        src_idx, dst_idx = valid_edges[int(idx)]
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
                correlation_score=0.0,
                rank=rank,
            )
        )
    return candidates


def propose_lowrank_edge_candidates(
    tensors: Sequence[PatchEffectTensor],
    num_edges: int,
    node_types: Sequence[str] = (NODE_TYPE_ATT,),
    enforce_direction: bool = True,
    eps: float = 1e-6,
) -> list[CausalEdgeCandidate]:
    """Propose low-ranked edge candidates (lowest correlation scores).

    Uses the same scoring as ``propose_causal_edge_candidates`` but
    returns the edges with the lowest correlation scores. This provides
    a negative-control baseline: edges the graph construction predicts
    should have the weakest co-variation.
    """
    if not tensors:
        return []

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
            if (
                src.node_type == NODE_TYPE_ATT
                and dst.node_type == NODE_TYPE_ATT
                and src.layer == dst.layer
            ):
                continue
            weight = float(corr[src_idx, dst_idx])
            if not np.isfinite(weight):
                continue
            scored.append((abs(weight), weight, src_idx, dst_idx))

    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    # Take lowest-ranked edges
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


def _candidate_from_nodes(
    *,
    rank: int,
    score: float,
    src: Node,
    dst: Node,
) -> CausalEdgeCandidate:
    return CausalEdgeCandidate(
        src_layer=src.layer,
        src_token=src.token,
        src_node_type=src.node_type,
        src_head=src.head,
        dst_layer=dst.layer,
        dst_token=dst.token,
        dst_node_type=dst.node_type,
        dst_head=dst.head,
        correlation_score=score,
        rank=rank,
    )


def _effect_feature_matrix(
    tensors: Sequence[PatchEffectTensor],
    *,
    eps: float,
) -> tuple[NDArray[np.float64], list[Node]]:
    component_axis = tensors[0].component_axis
    num_layers = tensors[0].num_layers
    min_tokens = min(t.num_tokens for t in tensors)
    num_components = len(component_axis)

    rows: list[NDArray[np.float64]] = []
    for tensor in tensors:
        delta = max(float(tensor.clean_score - tensor.base_score), eps)
        normalized = tensor.effects[:, :min_tokens, :] / delta
        rows.append(normalized.reshape(-1).astype(np.float64, copy=False))

    nodes = [
        Node.from_index(i, num_tokens=min_tokens, component_axis=component_axis)
        for i in range(num_layers * min_tokens * num_components)
    ]
    return np.stack(rows, axis=0), nodes


def _valid_edge_indices(
    nodes: Sequence[Node],
    *,
    node_types: Sequence[str],
    enforce_direction: bool,
) -> list[tuple[int, int]]:
    allowed_types = set(node_types)
    valid_nodes = [
        index for index, node in enumerate(nodes) if node.node_type in allowed_types
    ]
    edge_indices: list[tuple[int, int]] = []
    for src_idx in valid_nodes:
        src = nodes[src_idx]
        for dst_idx in valid_nodes:
            if src_idx == dst_idx:
                continue
            dst = nodes[dst_idx]
            if enforce_direction and not (src < dst):
                continue
            if (
                src.node_type == NODE_TYPE_ATT
                and dst.node_type == NODE_TYPE_ATT
                and src.layer == dst.layer
            ):
                continue
            edge_indices.append((src_idx, dst_idx))
    return edge_indices


def _ranked_edges_from_scores(
    scores: NDArray[np.float64],
    nodes: Sequence[Node],
    valid_edges: Sequence[tuple[int, int]],
    *,
    num_edges: int,
    reverse: bool,
    excluded_edge_ids: set[str] | None = None,
) -> list[CausalEdgeCandidate]:
    excluded = excluded_edge_ids or set()
    ranked: list[tuple[float, float, int, int]] = []
    for src_idx, dst_idx in valid_edges:
        score = float(scores[src_idx, dst_idx])
        if not np.isfinite(score):
            continue
        ranked.append((abs(score), score, src_idx, dst_idx))
    if reverse:
        ranked.sort(key=lambda item: (-item[0], -abs(item[1]), item[2], item[3]))
    else:
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    candidates: list[CausalEdgeCandidate] = []
    for _, score, src_idx, dst_idx in ranked:
        candidate = _candidate_from_nodes(
            rank=len(candidates) + 1,
            score=score,
            src=nodes[src_idx],
            dst=nodes[dst_idx],
        )
        if candidate.edge_id() in excluded:
            continue
        candidates.append(candidate)
        if len(candidates) >= num_edges:
            break
    return candidates


def propose_partial_correlation_edge_candidates(
    tensors: Sequence[PatchEffectTensor],
    num_edges: int = 5,
    node_types: Sequence[str] = (NODE_TYPE_RES,),
    enforce_direction: bool = True,
    eps: float = 1e-6,
    ridge: float = 1e-3,
) -> list[CausalEdgeCandidate]:
    """Propose candidate edges from ridge-regularized partial correlations."""
    if not tensors or num_edges <= 0:
        return []
    for node_type in node_types:
        _validate_node_type(node_type)

    matrix, nodes = _effect_feature_matrix(tensors, eps=eps)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    denom = max(centered.shape[0] - 1, 1)
    covariance = (centered.T @ centered) / denom
    scale = float(np.trace(covariance) / max(covariance.shape[0], 1))
    covariance = covariance + np.eye(covariance.shape[0]) * max(scale * ridge, ridge)
    precision = np.linalg.pinv(covariance)
    diag = np.sqrt(np.maximum(np.diag(precision), eps))
    partial = -precision / np.maximum(np.outer(diag, diag), eps)
    np.fill_diagonal(partial, 0.0)

    valid_edges = _valid_edge_indices(
        nodes,
        node_types=node_types,
        enforce_direction=enforce_direction,
    )
    return _ranked_edges_from_scores(
        partial,
        nodes,
        valid_edges,
        num_edges=num_edges,
        reverse=True,
    )


def propose_causal_candidate_groups(
    tensors: Sequence[PatchEffectTensor],
    num_edges: int = 5,
    node_types: Sequence[str] = (NODE_TYPE_RES,),
    enforce_direction: bool = True,
    eps: float = 1e-6,
    seed: int = 42,
) -> dict[str, list[CausalEdgeCandidate]]:
    """Return disjoint CI/PC/null candidate groups for DI validation."""
    if not tensors or num_edges <= 0:
        return {"top_ci": [], "top_pc": [], "random": [], "lowrank": []}

    for node_type in node_types:
        _validate_node_type(node_type)

    groups: dict[str, list[CausalEdgeCandidate]] = {}
    used: set[str] = set()

    top_ci = propose_causal_edge_candidates(
        tensors,
        num_edges=num_edges,
        node_types=node_types,
        enforce_direction=enforce_direction,
        eps=eps,
    )
    groups["top_ci"] = top_ci
    used.update(edge.edge_id() for edge in top_ci)

    pc_pool = propose_partial_correlation_edge_candidates(
        tensors,
        num_edges=max(num_edges * 4, num_edges),
        node_types=node_types,
        enforce_direction=enforce_direction,
        eps=eps,
    )
    groups["top_pc"] = [edge for edge in pc_pool if edge.edge_id() not in used][
        :num_edges
    ]
    used.update(edge.edge_id() for edge in groups["top_pc"])

    random_pool = propose_random_edge_candidates(
        tensors,
        num_edges=max(num_edges * 10, num_edges),
        node_types=node_types,
        enforce_direction=enforce_direction,
        seed=seed,
    )
    groups["random"] = [edge for edge in random_pool if edge.edge_id() not in used][
        :num_edges
    ]
    used.update(edge.edge_id() for edge in groups["random"])

    lowrank_pool = propose_lowrank_edge_candidates(
        tensors,
        num_edges=max(num_edges * 10, num_edges),
        node_types=node_types,
        enforce_direction=enforce_direction,
        eps=eps,
    )
    groups["lowrank"] = [edge for edge in lowrank_pool if edge.edge_id() not in used][
        :num_edges
    ]
    return groups


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
        distractor_token = get_prompt_pair_distractor(pair)
        clean_cache = model.cache_clean_components(pair.x_cln, node_types=required_node_types)
        base_cache = model.cache_clean_components(pair.x_crp, node_types=required_node_types)
        clean_score = float(
            model.score(
                pair.x_cln,
                pair.y_star,
                distractor_token=distractor_token,
            )
        )
        base_score = float(
            model.score(
                pair.x_crp,
                pair.y_star,
                distractor_token=distractor_token,
            )
        )
        example_contexts.append(
            {
                "pair": pair,
                "clean_cache": clean_cache,
                "base_cache": base_cache,
                "clean_score": clean_score,
                "base_score": base_score,
                "distractor_token": distractor_token,
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
                distractor_token = context["distractor_token"]

                base_v = _get_activation(base_cache, dst_node)
                clean_v = _get_activation(clean_cache, dst_node)
                if base_v is None or clean_v is None:
                    continue

                score_u, patched_capture = model.patched_score_multi_with_capture(
                    pair.x_crp,
                    pair.y_star,
                    patch_cache=clean_cache,
                    patch_nodes={src_node},
                    distractor_token=distractor_token,
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
                    distractor_token=distractor_token,
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
                    distractor_token=distractor_token,
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
                    distractor_token=distractor_token,
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

    multiple_testing = _apply_multiple_testing_corrections(edge_results, alpha=0.05)

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
        "multiple_testing": {
            "methods": ["fdr_bh", "bonferroni"],
            "alpha": 0.05,
            "metrics": multiple_testing,
        },
    }

    return CausalEvalResult(
        model_name=getattr(model, "model_name", "unknown"),
        config=config,
        run_metadata=metadata,
        edges=edge_results,
        arrays=arrays,
    )
