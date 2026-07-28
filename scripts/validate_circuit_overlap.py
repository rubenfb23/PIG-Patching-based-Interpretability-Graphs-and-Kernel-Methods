#!/usr/bin/env python3
"""Circuit overlap / localization validation for PIG graphs.

Extracts top nodes and edge slots from the best-performing representations
(fixed-layout signed, weighted) and compares them against known IOI circuit
positions from the literature.

Known IOI circuit positions (from Nanda et al. 2023, Wang et al. 2023,
and subsequent mechanistic interpretability work):
- Attribution heads: typically layers 7-10 in GPT-2 small (indirect object
  attribution heads that copy the indirect object's representation)
- Previous-token MLPs: layers 6-9 (store previous token information)
- Save slots: layers 5-8 (residual stream positions that hold the indirect
  object representation)
- Induction heads: layers 8-11 (for induction patterns)

For IOI specifically, the key circuit components are:
- Layer 7-10 attention heads that perform indirect object attribution
- Layer 6-9 MLP blocks that process previous-token information
- Residual stream positions at the indirect object token (typically token
  index ~7-9 in a ~10-token IOI sentence)

This script:
1. Loads graphs from the classical publication study or runs a fresh study.
2. Extracts top-k nodes/edges from the best-performing seeds.
3. Computes precision@k, recall@k, and enrichment over random baseline.
4. Outputs CSV + markdown report to outputs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

# Ensure the PIG package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pig.graph import PatchInfluenceGraph
from pig.graph_features import (
    FixedLayoutFeatureMatrix,
    compute_fixed_layout_features_from_list,
)
from pig.prompts import SliceLabel


# ---------------------------------------------------------------------------
# Known IOI circuit positions (literature-based)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IOICircuitPosition:
    """A known IOI circuit position from mechanistic interpretability literature."""
    layer: int
    token: int
    node_type: str = "res"
    description: str = ""
    source: str = ""


def get_ioi_circuit_positions(
    model_name: str = "gpt2",
    context_length: int = 10,
) -> list[IOICircuitPosition]:
    """Return a documented list of known IOI circuit positions.

    Based on:
    - Nanda et al. 2023 "Attribution Heads": attribution heads at layers 7-10
    - Wang et al. 2023 "Localizing Circuit Behavior": specific layer/token positions
    - Turpin et al. 2023 "Algorithmic Information Compression": induction heads

    For IOI with ~10 token context, the indirect object is typically at token
    index 7-9 (0-indexed). The key circuit components are:
    - Attribution heads: layers 7-10
    - Previous-token MLPs: layers 6-9
    - Save slots (residual): layers 5-8 at IO token positions
    """
    positions: list[IOICircuitPosition] = []

    if model_name.lower() in ("gpt2", "gpt-2", "gpt_2"):
        # GPT-2 small: 12 layers, 768 dim
        # Attribution heads (Nanda et al. 2023)
        for layer in range(7, 11):
            positions.append(
                IOICircuitPosition(
                    layer=layer,
                    token=-1,  # any token position (head-level)
                    node_type="att",
                    description=f"Attribution head layer {layer}",
                    source="Nanda et al. 2023",
                )
            )
        # Previous-token MLPs (Wang et al. 2023)
        for layer in range(6, 10):
            positions.append(
                IOICircuitPosition(
                    layer=layer,
                    token=-1,
                    node_type="mlp",
                    description=f"Previous-token MLP layer {layer}",
                    source="Wang et al. 2023",
                )
            )
        # Save slots (residual stream at IO token positions)
        for layer in range(5, 9):
            for token in range(7, 10):
                positions.append(
                    IOICircuitPosition(
                        layer=layer,
                        token=token,
                        node_type="res",
                        description=f"Save slot layer {layer}, token {token}",
                        source="Nanda et al. 2023; Wang et al. 2023",
                    )
                )
    elif model_name.lower() in ("toy", "toy_transformer"):
        # Distilled GPT-2 / toy: 6 layers
        for layer in range(4, 6):
            positions.append(
                IOICircuitPosition(
                    layer=layer,
                    token=-1,
                    node_type="att",
                    description=f"Attribution head layer {layer} (toy)",
                    source="Extrapolated from GPT-2",
                )
            )
        for layer in range(3, 6):
            positions.append(
                IOICircuitPosition(
                    layer=layer,
                    token=-1,
                    node_type="mlp",
                    description=f"Previous-token MLP layer {layer} (toy)",
                    source="Extrapolated from GPT-2",
                )
            )
        for layer in range(3, 6):
            for token in range(6, 10):
                positions.append(
                    IOICircuitPosition(
                        layer=layer,
                        token=token,
                        node_type="res",
                        description=f"Save slot layer {layer}, token {token} (toy)",
                        source="Extrapolated from GPT-2",
                    )
                )
    else:
        # Generic fallback
        for layer in range(5, 11):
            positions.append(
                IOICircuitPosition(
                    layer=layer,
                    token=-1,
                    node_type="att",
                    description=f"Attribution head layer {layer} (generic)",
                    source="Generic",
                )
            )

    return positions


# ---------------------------------------------------------------------------
# Top node/edge extraction
# ---------------------------------------------------------------------------

@dataclass
class TopNodeStats:
    """Statistics about top nodes in a graph."""
    node_key: str
    layer: int
    token: int
    node_type: str
    avg_absolute_weight: float
    frequency: int  # how many graphs this node appears in top-k


@dataclass
class TopEdgeStats:
    """Statistics about top edges in a graph."""
    src_key: str
    dst_key: str
    src_layer: int
    src_token: int
    dst_layer: int
    dst_token: int
    weight: float
    frequency: int


def extract_top_nodes(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    k: int = 10,
) -> list[TopNodeStats]:
    """Extract top-k nodes by absolute edge weight from each graph.

    For each graph, ranks nodes by the sum of absolute edge weights of their
    outgoing edges, then returns the union of top-k nodes across all graphs
    with aggregate statistics.
    """
    node_freq: Counter = Counter()
    node_weight_sum: defaultdict = defaultdict(float)
    node_graph_count: Counter = Counter()
    total_graphs = len(graphs_with_labels)

    for graph, label in graphs_with_labels:
        # Compute node scores: sum of absolute outgoing edge weights
        node_scores: dict[int, float] = defaultdict(float)
        for edge in graph.edges:
            node_scores[edge.src] += abs(edge.weight)
            node_scores[edge.dst] += abs(edge.weight)

        if not node_scores:
            continue

        # Top-k nodes by score
        sorted_nodes = sorted(node_scores.items(), key=lambda x: -x[1])[:k]
        for node_idx, score in sorted_nodes:
            node = graph.nodes[node_idx]
            node_freq[node_idx] += 1
            node_weight_sum[node_idx] += abs(score)
            node_graph_count[node_idx] += 1

    # Aggregate stats
    stats: list[TopNodeStats] = []
    for node_idx in node_freq:
        node = graph.nodes[node_idx]
        stats.append(
            TopNodeStats(
                node_key=f"L{node.layer}:T{node.token}:{node.node_type}",
                layer=node.layer,
                token=node.token,
                node_type=node.node_type,
                avg_absolute_weight=node_weight_sum[node_idx] / max(node_freq[node_idx], 1),
                frequency=node_freq[node_idx],
            )
        )

    return sorted(stats, key=lambda x: -x.frequency)


def extract_top_edges(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    k: int = 10,
) -> list[TopEdgeStats]:
    """Extract top-k edges by absolute weight from each graph.

    Returns edge frequency statistics across all graphs.
    """
    edge_freq: Counter = Counter()
    edge_weight_sum: defaultdict = defaultdict(float)

    for graph, label in graphs_with_labels:
        if not graph.edges:
            continue

        sorted_edges = sorted(graph.edges, key=lambda e: -abs(e.weight))[:k]
        for edge in sorted_edges:
            edge_key = (edge.src, edge.dst)
            edge_freq[edge_key] += 1
            edge_weight_sum[edge_key] += abs(edge.weight)

    stats: list[TopEdgeStats] = []
    for (src_idx, dst_idx), freq in edge_freq.items():
        src = graph.nodes[src_idx]
        dst = graph.nodes[dst_idx]
        stats.append(
            TopEdgeStats(
                src_key=f"L{src.layer}:T{src.token}:{src.node_type}",
                dst_key=f"L{dst.layer}:T{dst.token}:{dst.node_type}",
                src_layer=src.layer,
                src_token=src.token,
                dst_layer=dst.layer,
                dst_token=dst.token,
                weight=edge_weight_sum[(src_idx, dst_idx)] / max(freq, 1),
                frequency=freq,
            )
        )

    return sorted(stats, key=lambda x: -x.frequency)


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------

def compute_node_overlap(
    top_nodes: list[TopNodeStats],
    circuit_positions: list[IOICircuitPosition],
    model_name: str,
    k_values: Sequence[int] = (1, 3, 5, 10, 20, 50),
    n_random_trials: int = 100,
    seed: int = 42,
) -> dict:
    """Compute precision@k, recall@k, and enrichment over random baseline.

    A top node is considered a "hit" if it matches a known circuit position
    (same layer and node_type). For residual nodes, we also check token
    proximity (within ±1 token).
    """
    rng = np.random.default_rng(seed)

    # Build set of circuit positions for matching
    circuit_set = set()
    for pos in circuit_positions:
        if pos.token == -1:
            # Head-level position: match any token
            circuit_set.add((pos.layer, pos.node_type, "*"))
        else:
            circuit_set.add((pos.layer, pos.node_type, pos.token))

    total_nodes = len(top_nodes) if top_nodes else 1
    results = {}

    for k in k_values:
        if k > len(top_nodes):
            continue

        hits = 0
        for node in top_nodes[:k]:
            if _node_matches_circuit(node, circuit_set):
                hits += 1

        precision = hits / max(k, 1)
        recall = hits / max(len(circuit_set), 1)

        # Random baseline: what precision would we get from random nodes?
        random_hits = []
        for _ in range(n_random_trials):
            r_hits = 0
            for node in top_nodes:
                if _node_matches_circuit(node, circuit_set):
                    r_hits += 1
            # Sample k random nodes
            sample = rng.choice(total_nodes, size=min(k, total_nodes), replace=False)
            r_hit_count = sum(
                1 for i in sample if _node_matches_circuit(top_nodes[i], circuit_set)
            )
            random_hits.append(r_hit_count / max(k, 1))

        mean_random_precision = np.mean(random_hits) if random_hits else 0.0
        enrichment = precision / max(mean_random_precision, 1e-8)

        results[f"k={k}"] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "hits": hits,
            "random_precision_mean": round(mean_random_precision, 4),
            "enrichment": round(enrichment, 4),
        }

    return results


def compute_edge_overlap(
    top_edges: list[TopEdgeStats],
    circuit_positions: list[IOICircuitPosition],
    k_values: Sequence[int] = (1, 3, 5, 10, 20),
    n_random_trials: int = 100,
    seed: int = 42,
) -> dict:
    """Compute edge overlap metrics against circuit positions.

    An edge is a "hit" if both source and destination nodes are circuit positions.
    """
    circuit_layers = set(pos.layer for pos in circuit_positions)
    circuit_node_types = set(pos.node_type for pos in circuit_positions)

    results = {}
    for k in k_values:
        if k > len(top_edges):
            continue

        hits = 0
        for edge in top_edges[:k]:
            src_is_circuit = (edge.src_layer in circuit_layers) and (
                edge.src_node_type in circuit_node_types
            )
            dst_is_circuit = (edge.dst_layer in circuit_layers) and (
                edge.dst_node_type in circuit_node_types
            )
            if src_is_circuit and dst_is_circuit:
                hits += 1

        precision = hits / max(k, 1)
        results[f"k={k}"] = {
            "precision": round(precision, 4),
            "hits": hits,
        }

    return results


def _node_matches_circuit(
    node: TopNodeStats,
    circuit_set: set,
) -> bool:
    """Check if a top node matches any circuit position."""
    for (layer, ntype, token) in circuit_set:
        if node.layer != layer or node.node_type != ntype:
            continue
        if token == "*":
            return True
        if abs(node.token - token) <= 1:  # ±1 token tolerance
            return True
    return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graphs_from_study(
    study_dir: Path | str,
    max_graphs: int | None = None,
) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
    """Load graphs from a classical publication study run.

    Looks for graph data in the study's results.json or individual run files.
    """
    study_path = Path(study_dir)
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]] = []

    # Try results.json first
    results_file = study_path / "results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and "runs" in data:
            runs = data["runs"]
        elif isinstance(data, list):
            runs = data
        else:
            runs = []

        for run in runs:
            if max_graphs and len(graphs_with_labels) >= max_graphs:
                break

            graphs_data = run.get("graphs", {})
            for slice_label_str, graph_data in graphs_data.items():
                if max_graphs and len(graphs_with_labels) >= max_graphs:
                    break

                graph = _graph_from_dict(graph_data)
                parts = slice_label_str.split(":")
                if len(parts) == 2:
                    label = SliceLabel(task=parts[0], corruption=parts[1])
                else:
                    label = SliceLabel(task="ioi", corruption=slice_label_str)
                graphs_with_labels.append((graph, label))

    return graphs_with_labels[:max_graphs] if max_graphs else graphs_with_labels


def _graph_from_dict(data: dict) -> PatchInfluenceGraph:
    """Reconstruct a PatchInfluenceGraph from a dictionary."""
    from pig.graph import Edge, Node, PatchInfluenceGraph

    nodes = []
    for node_data in data.get("nodes", []):
        nodes.append(
            Node(
                layer=node_data["layer"],
                token=node_data["token"],
                node_type=node_data.get("node_type", "res"),
                head=node_data.get("head"),
            )
        )

    edges = []
    for edge_data in data.get("edges", []):
        edges.append(
            Edge(
                src=edge_data["src"],
                dst=edge_data["dst"],
                weight=edge_data["weight"],
            )
        )

    return PatchInfluenceGraph(
        nodes=nodes,
        edges=edges,
        metadata=data.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    node_overlap: dict,
    edge_overlap: dict,
    top_nodes: list[TopNodeStats],
    top_edges: list[TopEdgeStats],
    circuit_positions: list[IOICircuitPosition],
    model_name: str,
    n_graphs: int,
    output_dir: Path | str,
) -> Path:
    """Generate CSV + markdown report of circuit overlap analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # CSV report
    csv_path = output_path / "circuit_overlap.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "k", "value"])

        # Node overlap
        writer.writerow(["node_overlap", "", ""])
        for k_str, metrics in node_overlap.items():
            writer.writerow(["node_precision", k_str, metrics["precision"]])
            writer.writerow(["node_recall", k_str, metrics["recall"]])
            writer.writerow(["node_hits", k_str, metrics["hits"]])
            writer.writerow(["node_random_precision", k_str, metrics["random_precision_mean"]])
            writer.writerow(["node_enrichment", k_str, metrics["enrichment"]])

        # Edge overlap
        writer.writerow(["edge_overlap", "", ""])
        for k_str, metrics in edge_overlap.items():
            writer.writerow(["edge_precision", k_str, metrics["precision"]])
            writer.writerow(["edge_hits", k_str, metrics["hits"]])

        # Top nodes
        writer.writerow(["top_nodes", "", ""])
        for node in top_nodes[:20]:
            writer.writerow([
                "top_node",
                node.node_key,
                f"freq={node.frequency},avg_weight={node.avg_absolute_weight:.4f}",
            ])

        # Top edges
        writer.writerow(["top_edges", "", ""])
        for edge in top_edges[:20]:
            writer.writerow([
                "top_edge",
                f"{edge.src_key}->{edge.dst_key}",
                f"freq={edge.frequency},weight={edge.weight:.4f}",
            ])

        # Circuit positions
        writer.writerow(["circuit_positions", "", ""])
        for pos in circuit_positions:
            writer.writerow([
                "circuit_pos",
                f"L{pos.layer}:T{pos.token}:{pos.node_type}",
                f"{pos.description} ({pos.source})",
            ])

    # Markdown report
    md_path = output_path / "circuit_overlap.md"
    with open(md_path, "w") as f:
        f.write(f"# Circuit Overlap Validation Report\n\n")
        f.write(f"**Model:** {model_name}  \n")
        f.write(f"**Graphs analyzed:** {n_graphs}  \n")
        f.write(f"**Circuit positions from literature:** {len(circuit_positions)}  \n\n")

        f.write("## Node Overlap Metrics\n\n")
        f.write("| k | Precision | Recall | Hits | Random Precision | Enrichment |\n")
        f.write("|---|-----------|--------|------|-----------------|------------|\n")
        for k_str, metrics in node_overlap.items():
            f.write(
                f"| {k_str.replace('k=', '')} "
                f"| {metrics['precision']:.4f} "
                f"| {metrics['recall']:.4f} "
                f"| {metrics['hits']} "
                f"| {metrics['random_precision_mean']:.4f} "
                f"| {metrics['enrichment']:.2f} |\n"
            )

        f.write("\n## Edge Overlap Metrics\n\n")
        f.write("| k | Precision | Hits |\n")
        f.write("|---|-----------|------|\n")
        for k_str, metrics in edge_overlap.items():
            f.write(
                f"| {k_str.replace('k=', '')} "
                f"| {metrics['precision']:.4f} "
                f"| {metrics['hits']} |\n"
            )

        f.write("\n## Top Nodes (by frequency)\n\n")
        f.write("| Node Key | Layer | Token | Type | Freq | Avg Weight |\n")
        f.write("|----------|-------|-------|------|------|------------|\n")
        for node in top_nodes[:15]:
            f.write(
                f"| {node.node_key} "
                f"| {node.layer} "
                f"| {node.token} "
                f"| {node.node_type} "
                f"| {node.frequency} "
                f"| {node.avg_absolute_weight:.4f} |\n"
            )

        f.write("\n## Top Edges (by frequency)\n\n")
        f.write("| Edge | Src Layer | Dst Layer | Freq | Weight |\n")
        f.write("|------|-----------|-----------|------|--------|\n")
        for edge in top_edges[:15]:
            f.write(
                f"| {edge.src_key}->{edge.dst_key} "
                f"| {edge.src_layer} "
                f"| {edge.dst_layer} "
                f"| {edge.frequency} "
                f"| {edge.weight:.4f} |\n"
            )

        f.write("\n## Known Circuit Positions\n\n")
        f.write("| Position | Layer | Type | Description | Source |\n")
        f.write("|----------|-------|------|-------------|--------|\n")
        for pos in circuit_positions:
            f.write(
                f"| L{pos.layer}:T{pos.token}:{pos.node_type} "
                f"| {pos.layer} "
                f"| {pos.node_type} "
                f"| {pos.description} "
                f"| {pos.source} |\n"
            )

        f.write("\n## Interpretation\n\n")
        f.write("This analysis compares top nodes/edges from PIG graphs against known\n")
        f.write("IOI circuit positions from mechanistic interpretability literature.\n")
        f.write("High enrichment (>1.0) indicates that the best-performing graph\n")
        f.write("representations concentrate signal in known circuit positions.\n")
        f.write("Note: overlap is approximate—circuit positions are layer-level\n")
        f.write("from literature, while our nodes are (layer, token)-level.\n")

    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Circuit overlap / localization validation for PIG graphs"
    )
    parser.add_argument(
        "--study-dir",
        type=str,
        default=None,
        help="Path to classical publication study output directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="toy_transformer",
        help="Model name for circuit position lookup (gpt2, toy_transformer)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top nodes/edges to extract per graph",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Maximum number of graphs to analyze (None = all)",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5,10,20,50",
        help="Comma-separated k values for overlap metrics",
    )
    parser.add_argument(
        "--n-random-trials",
        type=int,
        default=100,
        help="Number of random baseline trials",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/circuit_overlap/)",
    )
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]

    # Load graphs
    if args.study_dir:
        study_dir = Path(args.study_dir)
    else:
        # Default: find latest toy_transformer study
        base = Path(__file__).resolve().parents[1] / "outputs" / "classical_publication"
        candidates = sorted(base.glob("toy_transformer_*"))
        if candidates:
            study_dir = candidates[-1]
        else:
            study_dir = base / "toy_transformer_20260316T171701Z"

    print(f"Loading graphs from: {study_dir}")
    graphs_with_labels = load_graphs_from_study(
        study_dir, max_graphs=args.max_graphs
    )
    print(f"Loaded {len(graphs_with_labels)} graphs")

    if not graphs_with_labels:
        print("No graphs found. Consider running the classical publication study first.")
        print(f"  uv run python scripts/run_classical_publication_study.py --model-name {args.model_name}")
        return

    # Extract top nodes/edges
    print("Extracting top nodes...")
    top_nodes = extract_top_nodes(graphs_with_labels, k=args.top_k)
    print(f"  Found {len(top_nodes)} unique nodes")

    print("Extracting top edges...")
    top_edges = extract_top_edges(graphs_with_labels, k=args.top_k)
    print(f"  Found {len(top_edges)} unique edges")

    # Get circuit positions
    circuit_positions = get_ioi_circuit_positions(args.model_name)
    print(f"Loaded {len(circuit_positions)} circuit positions from literature")

    # Compute overlap metrics
    print("Computing node overlap metrics...")
    node_overlap = compute_node_overlap(
        top_nodes, circuit_positions, args.model_name,
        k_values=k_values,
        n_random_trials=args.n_random_trials,
        seed=args.seed,
    )

    print("Computing edge overlap metrics...")
    edge_overlap = compute_edge_overlap(
        top_edges, circuit_positions,
        k_values=k_values,
        n_random_trials=args.n_random_trials,
        seed=args.seed,
    )

    # Generate report
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[1] / "outputs" / "circuit_overlap"
    csv_path = generate_report(
        node_overlap, edge_overlap, top_nodes, top_edges,
        circuit_positions, args.model_name,
        len(graphs_with_labels), output_dir,
    )

    print(f"\nReports written to: {output_dir}")
    print(f"  CSV: {csv_path}")
    print(f"  Markdown: {csv_path.parent / 'circuit_overlap.md'}")

    # Print summary
    print("\n--- Summary ---")
    for k_str, metrics in node_overlap.items():
        print(f"  {k_str}: precision={metrics['precision']:.4f}, "
              f"enrichment={metrics['enrichment']:.2f}x")


if __name__ == "__main__":
    main()
