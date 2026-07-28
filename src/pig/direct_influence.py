"""Exhaustive head-to-head causal graph construction and comparison.

The routines in this module evaluate every temporally valid attention-head edge
in a selected layer/head universe.  They deliberately avoid candidate
screening: proposal, CI, and path-patching scores share the exact same edge
universe.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from pig.causal import restoration_fraction
from pig.model import ActivationCache, HookedModel, NODE_TYPE_ATT
from pig.prompts import PromptPair, get_prompt_pair_distractor


@dataclass(frozen=True, order=True)
class HeadNode:
    """Attention-head node aggregated over all shared prompt positions."""

    layer: int
    head: int

    def label(self) -> str:
        return f"L{self.layer}H{self.head}"


@dataclass(frozen=True)
class HeadEdge:
    """Directed edge between heads in strictly increasing layer order."""

    source: HeadNode
    target: HeadNode

    def __post_init__(self) -> None:
        if self.source.layer >= self.target.layer:
            raise ValueError("Head edges must point to a strictly later layer")

    def edge_id(self) -> str:
        return f"{self.source.label()}->{self.target.label()}"


@dataclass(frozen=True)
class FullGraphConfig:
    """Configuration shared by exhaustive graph evaluation and aggregation."""

    eps: float = 1e-6
    bootstrap_samples: int = 200
    ci_alpha: float = 0.05
    active_threshold: float = 0.0
    edge_budget: int = 256
    seed: int = 42


@dataclass
class PromptGraphScores:
    """Per-prompt scores over one fixed exhaustive edge universe."""

    edge_ids: list[str]
    direct_influence: NDArray[np.float64]
    causal_intervention: NDArray[np.float64]
    path_patching: NDArray[np.float64]
    metadata: dict

    def save(self, path: Path | str) -> Path:
        """Save a restartable prompt shard."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            edge_ids=np.asarray(self.edge_ids, dtype=np.str_),
            direct_influence=self.direct_influence,
            causal_intervention=self.causal_intervention,
            path_patching=self.path_patching,
            metadata_json=np.asarray(json.dumps(self.metadata), dtype=np.str_),
        )
        return output_path

    @classmethod
    def load(cls, path: Path | str) -> "PromptGraphScores":
        """Load a prompt shard written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as payload:
            return cls(
                edge_ids=payload["edge_ids"].astype(str).tolist(),
                direct_influence=np.asarray(
                    payload["direct_influence"], dtype=np.float64
                ),
                causal_intervention=np.asarray(
                    payload["causal_intervention"], dtype=np.float64
                ),
                path_patching=np.asarray(payload["path_patching"], dtype=np.float64),
                metadata=json.loads(str(payload["metadata_json"].item())),
            )


def enumerate_head_nodes(
    n_layers: int,
    n_heads: int,
    *,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
) -> list[HeadNode]:
    """Return the selected attention-head universe in stable order."""
    selected_layers = list(range(n_layers)) if layers is None else sorted(set(layers))
    selected_heads = list(range(n_heads)) if heads is None else sorted(set(heads))
    if not selected_layers or not selected_heads:
        raise ValueError("At least one layer and one head must be selected")
    if any(layer < 0 or layer >= n_layers for layer in selected_layers):
        raise ValueError(f"Layer selection must be within [0, {n_layers - 1}]")
    if any(head < 0 or head >= n_heads for head in selected_heads):
        raise ValueError(f"Head selection must be within [0, {n_heads - 1}]")
    return [
        HeadNode(layer=layer, head=head)
        for layer in selected_layers
        for head in selected_heads
    ]


def enumerate_forward_edges(nodes: Sequence[HeadNode]) -> list[HeadEdge]:
    """Enumerate every cross-layer edge without screening."""
    return [
        HeadEdge(source=source, target=target)
        for source in nodes
        for target in nodes
        if source.layer < target.layer
    ]


def _head_patch_nodes(node: HeadNode, num_tokens: int) -> set[tuple]:
    return {
        (node.layer, token, NODE_TYPE_ATT, node.head) for token in range(num_tokens)
    }


def _head_activation(
    cache: ActivationCache,
    node: HeadNode,
    num_tokens: int,
) -> NDArray[np.float64] | None:
    values = [
        cache.get(
            node.layer,
            token,
            node_type=NODE_TYPE_ATT,
            head=node.head,
        )
        for token in range(num_tokens)
    ]
    if any(value is None for value in values):
        return None
    return np.concatenate(
        [value.detach().float().cpu().numpy().reshape(-1) for value in values],
        axis=0,
    ).astype(np.float64, copy=False)


def _normalized_displacement(
    patched: NDArray[np.float64],
    baseline: NDArray[np.float64],
    clean: NDArray[np.float64],
    *,
    eps: float,
) -> float:
    numerator = float(np.linalg.norm(patched - baseline))
    denominator = float(np.linalg.norm(clean - baseline)) + eps
    return numerator / denominator


def evaluate_prompt_full_graph(
    model: HookedModel,
    prompt_pair: PromptPair,
    nodes: Sequence[HeadNode],
    edges: Sequence[HeadEdge],
    *,
    eps: float = 1e-6,
    show_progress: bool = False,
) -> PromptGraphScores:
    """Evaluate DI, CI, and path patching on every supplied edge for one prompt.

    Direct influence (DI)
        Patch all positions of sender ``u`` from clean into corrupted, then
        measure normalized activation displacement at receiver ``v``.

    Causal intervention (CI)
        Measure how much sender restoration is lost when receiver ``v`` is
        clamped to its corrupted activation: ``R(u) - R(u, clamp(v=base))``.

    Path patching (PC)
        Capture receiver ``v`` under the sender intervention, transplant only
        that induced receiver activation into the corrupted run, and measure
        output restoration.
    """
    if not nodes:
        raise ValueError("nodes must not be empty")
    if not edges:
        raise ValueError("edges must not be empty")

    node_set = set(nodes)
    if any(
        edge.source not in node_set or edge.target not in node_set for edge in edges
    ):
        raise ValueError("Every edge endpoint must belong to nodes")

    clean_tokens = model.get_num_tokens(prompt_pair.x_cln)
    corrupt_tokens = model.get_num_tokens(prompt_pair.x_crp)
    num_tokens = min(clean_tokens, corrupt_tokens)
    if num_tokens <= 0:
        raise ValueError("Prompt pair produced no shared token positions")

    distractor = get_prompt_pair_distractor(prompt_pair)
    clean_cache = model.cache_clean_components(
        prompt_pair.x_cln,
        node_types=[NODE_TYPE_ATT],
    )
    base_cache = model.cache_clean_components(
        prompt_pair.x_crp,
        node_types=[NODE_TYPE_ATT],
    )
    clean_score = float(
        model.score(
            prompt_pair.x_cln,
            prompt_pair.y_star,
            distractor_token=distractor,
        )
    )
    base_score = float(
        model.score(
            prompt_pair.x_crp,
            prompt_pair.y_star,
            distractor_token=distractor,
        )
    )

    edge_lookup: dict[HeadNode, list[tuple[int, HeadEdge]]] = {}
    for edge_index, edge in enumerate(edges):
        edge_lookup.setdefault(edge.source, []).append((edge_index, edge))

    direct = np.full(len(edges), np.nan, dtype=np.float64)
    intervention = np.full(len(edges), np.nan, dtype=np.float64)
    path = np.full(len(edges), np.nan, dtype=np.float64)

    base_activations = {
        node: _head_activation(base_cache, node, num_tokens) for node in nodes
    }
    clean_activations = {
        node: _head_activation(clean_cache, node, num_tokens) for node in nodes
    }

    iterator: Iterable[HeadNode] = [node for node in nodes if node in edge_lookup]
    progress = None
    if show_progress:
        from tqdm.auto import tqdm

        progress = tqdm(
            iterator,
            total=len(edge_lookup),
            desc="full-head-graph",
            dynamic_ncols=True,
        )
        iterator = progress

    try:
        for source in iterator:
            source_nodes = _head_patch_nodes(source, num_tokens)
            outgoing = edge_lookup[source]
            capture_nodes: set[tuple] = set()
            for _, edge in outgoing:
                capture_nodes.update(_head_patch_nodes(edge.target, num_tokens))

            source_score, induced_cache = model.patched_score_multi_with_capture(
                prompt_pair.x_crp,
                prompt_pair.y_star,
                patch_cache=clean_cache,
                patch_nodes=source_nodes,
                distractor_token=distractor,
                capture_nodes=capture_nodes,
            )
            source_restoration = restoration_fraction(
                source_score,
                base_score=base_score,
                clean_score=clean_score,
                eps=eps,
            )

            for edge_index, edge in outgoing:
                target_nodes = _head_patch_nodes(edge.target, num_tokens)
                base_target = base_activations[edge.target]
                clean_target = clean_activations[edge.target]
                induced_target = _head_activation(
                    induced_cache,
                    edge.target,
                    num_tokens,
                )
                if (
                    base_target is None
                    or clean_target is None
                    or induced_target is None
                ):
                    continue

                direct[edge_index] = _normalized_displacement(
                    induced_target,
                    base_target,
                    clean_target,
                    eps=eps,
                )

                blocked_score, _ = model.patched_score_multi_with_capture(
                    prompt_pair.x_crp,
                    prompt_pair.y_star,
                    patch_cache=clean_cache,
                    patch_nodes=source_nodes,
                    distractor_token=distractor,
                    clamp_cache=base_cache,
                    clamp_nodes=target_nodes,
                )
                blocked_restoration = restoration_fraction(
                    blocked_score,
                    base_score=base_score,
                    clean_score=clean_score,
                    eps=eps,
                )
                intervention[edge_index] = source_restoration - blocked_restoration

                path_score = model.patched_score_multi(
                    prompt_pair.x_crp,
                    prompt_pair.y_star,
                    induced_cache,
                    target_nodes,
                    distractor_token=distractor,
                )
                path[edge_index] = restoration_fraction(
                    path_score,
                    base_score=base_score,
                    clean_score=clean_score,
                    eps=eps,
                )
    finally:
        if progress is not None:
            progress.close()

    return PromptGraphScores(
        edge_ids=[edge.edge_id() for edge in edges],
        direct_influence=direct,
        causal_intervention=intervention,
        path_patching=path,
        metadata={
            "slice": str(prompt_pair.slice_label),
            "clean_prompt": prompt_pair.x_cln,
            "corrupted_prompt": prompt_pair.x_crp,
            "target": prompt_pair.y_star,
            "clean_score": clean_score,
            "base_score": base_score,
            "clean_tokens": clean_tokens,
            "corrupted_tokens": corrupt_tokens,
            "shared_tokens": num_tokens,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
        },
    )


def _bootstrap_edge_means(
    values: NDArray[np.float64],
    *,
    samples: int,
    alpha: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Bootstrap per-edge means while retaining NaN-safe edge coverage."""
    if values.ndim != 2:
        raise ValueError("values must have shape [prompts, edges]")
    means = np.nanmean(values, axis=0)
    if samples <= 0 or values.shape[0] <= 1:
        return means, means.copy(), means.copy()

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.shape[0], size=(samples, values.shape[0]))
    boot = np.nanmean(values[draws, :], axis=1)
    low = np.nanquantile(boot, alpha / 2.0, axis=0)
    high = np.nanquantile(boot, 1.0 - alpha / 2.0, axis=0)
    return means, low, high


def active_edges_from_lower_bound(
    lower_bound: NDArray[np.float64],
    *,
    threshold: float,
) -> NDArray[np.bool_]:
    """Mark edges whose lower confidence bound exceeds a causal threshold."""
    return np.isfinite(lower_bound) & (lower_bound > threshold)


def top_edge_mask(
    scores: NDArray[np.float64],
    *,
    budget: int,
) -> NDArray[np.bool_]:
    """Select the strongest positive edges for density-matched sensitivity."""
    mask = np.zeros(scores.shape, dtype=np.bool_)
    if budget <= 0:
        return mask
    valid = np.flatnonzero(np.isfinite(scores) & (scores > 0.0))
    if valid.size == 0:
        return mask
    ranked = valid[np.argsort(-scores[valid], kind="stable")]
    mask[ranked[: min(budget, ranked.size)]] = True
    return mask


def graph_alignment_metrics(
    predicted: NDArray[np.bool_],
    reference: NDArray[np.bool_],
    *,
    predicted_weights: NDArray[np.float64] | None = None,
    reference_weights: NDArray[np.float64] | None = None,
) -> dict[str, float | int]:
    """Compute binary edge overlap and optional weighted kernel alignment."""
    predicted = np.asarray(predicted, dtype=np.bool_)
    reference = np.asarray(reference, dtype=np.bool_)
    if predicted.shape != reference.shape:
        raise ValueError("predicted and reference masks must have identical shape")

    true_positive = int(np.sum(predicted & reference))
    false_positive = int(np.sum(predicted & ~reference))
    false_negative = int(np.sum(~predicted & reference))
    union = true_positive + false_positive + false_negative
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    jaccard = true_positive / union if union else 1.0
    edit_count = false_positive + false_negative
    result: dict[str, float | int] = {
        "predicted_edges": int(np.sum(predicted)),
        "reference_edges": int(np.sum(reference)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
        "graph_edit_distance": edit_count,
        "normalized_graph_edit_distance": (
            float(edit_count / predicted.size) if predicted.size else 0.0
        ),
    }

    if predicted_weights is not None and reference_weights is not None:
        first = np.nan_to_num(
            np.abs(np.asarray(predicted_weights, dtype=np.float64)),
            nan=0.0,
        )
        second = np.nan_to_num(
            np.abs(np.asarray(reference_weights, dtype=np.float64)),
            nan=0.0,
        )
        if first.shape != predicted.shape or second.shape != reference.shape:
            raise ValueError("Weight vectors must match their graph masks")
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        result["kernel_alignment"] = (
            float(np.dot(first, second) / denominator) if denominator > 0.0 else 0.0
        )
    return result


def aggregate_full_graph_scores(
    prompt_scores: Sequence[PromptGraphScores],
    nodes: Sequence[HeadNode],
    edges: Sequence[HeadEdge],
    config: FullGraphConfig,
) -> dict:
    """Aggregate prompt shards and build exhaustive graph comparisons."""
    if not prompt_scores:
        raise ValueError("prompt_scores must not be empty")
    expected_ids = [edge.edge_id() for edge in edges]
    if any(scores.edge_ids != expected_ids for scores in prompt_scores):
        raise ValueError("Prompt shards do not share the requested edge universe")

    matrices = {
        "direct_influence": np.stack(
            [scores.direct_influence for scores in prompt_scores], axis=0
        ),
        "causal_intervention": np.stack(
            [scores.causal_intervention for scores in prompt_scores], axis=0
        ),
        "path_patching": np.stack(
            [scores.path_patching for scores in prompt_scores], axis=0
        ),
    }
    summaries: dict[str, dict[str, NDArray]] = {}
    statistical_masks: dict[str, NDArray[np.bool_]] = {}
    budget_masks: dict[str, NDArray[np.bool_]] = {}
    for method_index, (method, matrix) in enumerate(matrices.items()):
        mean, low, high = _bootstrap_edge_means(
            matrix,
            samples=config.bootstrap_samples,
            alpha=config.ci_alpha,
            seed=config.seed + method_index * 10_000,
        )
        summaries[method] = {"mean": mean, "ci_low": low, "ci_high": high}
        statistical_masks[method] = active_edges_from_lower_bound(
            low,
            threshold=config.active_threshold,
        )
        budget_masks[method] = top_edge_mask(mean, budget=config.edge_budget)

    comparisons = [
        ("direct_influence", "causal_intervention"),
        ("path_patching", "causal_intervention"),
        ("causal_intervention", "direct_influence"),
        ("path_patching", "direct_influence"),
    ]

    def compare(mask_by_method: dict[str, NDArray[np.bool_]]) -> list[dict]:
        rows = []
        for predicted_name, reference_name in comparisons:
            metrics = graph_alignment_metrics(
                mask_by_method[predicted_name],
                mask_by_method[reference_name],
                predicted_weights=summaries[predicted_name]["mean"],
                reference_weights=summaries[reference_name]["mean"],
            )
            rows.append(
                {
                    "method": predicted_name,
                    "reference": reference_name,
                    **metrics,
                }
            )
        return rows

    edge_rows = []
    for edge_index, edge in enumerate(edges):
        row = {
            "edge_id": edge.edge_id(),
            "src_layer": edge.source.layer,
            "src_head": edge.source.head,
            "dst_layer": edge.target.layer,
            "dst_head": edge.target.head,
        }
        for method in matrices:
            row[f"{method}_mean"] = float(summaries[method]["mean"][edge_index])
            row[f"{method}_ci_low"] = float(summaries[method]["ci_low"][edge_index])
            row[f"{method}_ci_high"] = float(summaries[method]["ci_high"][edge_index])
            row[f"{method}_active"] = bool(statistical_masks[method][edge_index])
            row[f"{method}_top_budget"] = bool(budget_masks[method][edge_index])
        edge_rows.append(row)

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "metadata": {
            "num_prompts": len(prompt_scores),
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "full_edge_universe": True,
            "node_definition": "attention head aggregated across shared prompt tokens",
            "edge_constraint": "source.layer < target.layer",
            "methods": {
                "direct_influence": (
                    "normalized receiver activation displacement after sender patch"
                ),
                "causal_intervention": (
                    "sender restoration minus sender restoration with receiver clamped"
                ),
                "path_patching": (
                    "output restoration after transplanting sender-induced receiver"
                ),
            },
            "prompt_metadata": [scores.metadata for scores in prompt_scores],
        },
        "nodes": [asdict(node) | {"label": node.label()} for node in nodes],
        "summary_table": compare(statistical_masks),
        "top_budget_sensitivity": compare(budget_masks),
        "edge_rows": edge_rows,
        "_arrays": {
            **{
                f"{method}_prompt_scores": matrix for method, matrix in matrices.items()
            },
            **{
                f"{method}_{statistic}": values
                for method, method_summary in summaries.items()
                for statistic, values in method_summary.items()
            },
            **{f"{method}_active": mask for method, mask in statistical_masks.items()},
            **{f"{method}_top_budget": mask for method, mask in budget_masks.items()},
        },
    }


def _json_ready(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "_arrays"}


def save_full_graph_result(result: dict, output_dir: Path | str) -> dict[str, Path]:
    """Write JSON, NPZ, edge CSV, and the requested Markdown summary table."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    json_path = destination / "full_graph_comparison.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(result), handle, indent=2, allow_nan=False)

    npz_path = destination / "full_graph_comparison.npz"
    np.savez_compressed(npz_path, **result["_arrays"])

    edge_csv_path = destination / "full_graph_edges.csv"
    edge_rows = result["edge_rows"]
    with open(edge_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(edge_rows[0]))
        writer.writeheader()
        writer.writerows(edge_rows)

    summary_csv_path = destination / "summary_table.csv"
    summary_rows = result["summary_table"]
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_md_path = destination / "summary_table.md"
    lines = [
        "# Full Direct-Influence Graph Comparison",
        "",
        (
            f"Prompts: {result['metadata']['num_prompts']} | "
            f"Nodes: {result['metadata']['num_nodes']} | "
            f"Exhaustive edges: {result['metadata']['num_edges']}"
        ),
        "",
        "| Method | Reference | Precision | Recall | F1 | Jaccard | "
        "Norm. GED | Kernel alignment |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['reference']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['jaccard']:.4f} | "
            f"{row['normalized_graph_edit_distance']:.4f} | "
            f"{row['kernel_alignment']:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                "Active edges use the lower bootstrap confidence bound > "
                f"{result['config']['active_threshold']:.4g}. "
                "See `top_budget_sensitivity` in JSON for density-matched results."
            ),
            "",
        ]
    )
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "json": json_path,
        "npz": npz_path,
        "edges_csv": edge_csv_path,
        "summary_csv": summary_csv_path,
        "summary_markdown": summary_md_path,
    }
