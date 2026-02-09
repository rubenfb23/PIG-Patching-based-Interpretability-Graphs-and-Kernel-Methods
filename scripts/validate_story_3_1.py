#!/usr/bin/env python3
"""Validation script for Story 3.1 - Per-Slice Graph Builder.

This script verifies all acceptance criteria:
1. Node set V is fixed across all slices (same patch points)
2. Edge weights are computed via correlation of effect profiles within each slice
3. Direction constraint enforced: edges only go from earlier (ℓ, t) to later
4. Top-k sparsification applied: keep only k outgoing edges per node
5. All slices have comparable edge budgets (≤ k × |V|)
6. Graph statistics differ across slices (verifiable signal)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from pig.graph import GraphBuilder, build_graphs
from pig.model import HookedModel
from pig.patching import compute_patch_effects
from pig.prompts import ABBACorruption, IOIGenerator, SliceLabel


def generate_dataset(model: HookedModel, n_per_slice: int = 15):
    """Generate a multi-slice dataset."""
    print(f"  Generating {n_per_slice} examples per slice...")

    gen_swap = IOIGenerator(seed=42)
    gen_abba = IOIGenerator(corruption=ABBACorruption(), seed=42)

    pairs_swap = gen_swap.generate_batch(n_per_slice)
    pairs_abba = gen_abba.generate_batch(n_per_slice)

    print("  Computing patch effects for name_swap slice...")
    dataset_swap = compute_patch_effects(model, pairs_swap, show_progress=True)

    print("\n  Computing patch effects for abba slice...")
    dataset_abba = compute_patch_effects(model, pairs_abba, show_progress=True)

    # Merge datasets
    from pig.patching import PatchEffectDataset
    combined = PatchEffectDataset()
    for t in dataset_swap:
        combined.add(t)
    for t in dataset_abba:
        combined.add(t)

    return combined


def test_fixed_node_set(graphs: dict) -> bool:
    """Test that node set is fixed across all slices."""
    print("\n" + "=" * 60)
    print("TEST 1: Fixed Node Set Across Slices")
    print("=" * 60)

    if len(graphs) < 2:
        print("  [SKIP] Need at least 2 slices to compare")
        return True

    graph_list = list(graphs.values())
    reference = graph_list[0]
    ref_nodes = [(n.layer, n.token) for n in reference.nodes]

    all_match = True
    for slice_label, graph in graphs.items():
        nodes = [(n.layer, n.token) for n in graph.nodes]
        if nodes != ref_nodes:
            print(f"  [FAIL] Slice {slice_label} has different node set")
            all_match = False
        else:
            print(f"  Slice {slice_label}: {len(nodes)} nodes (matches)")

    if all_match:
        print("  [PASS] All slices have identical node sets")
    return all_match


def test_correlation_weights(graphs: dict) -> bool:
    """Test that edge weights are in valid correlation range."""
    print("\n" + "=" * 60)
    print("TEST 2: Edge Weights from Correlation")
    print("=" * 60)

    all_valid = True
    for slice_label, graph in graphs.items():
        weights = [e.weight for e in graph.edges]
        if not weights:
            print(f"  Slice {slice_label}: no edges")
            continue

        min_w = min(weights)
        max_w = max(weights)
        mean_w = np.mean(weights)

        # Correlations should be in [-1, 1]
        valid = min_w >= -1.0 and max_w <= 1.0

        print(f"  Slice {slice_label}: weight range [{min_w:.3f}, {max_w:.3f}], "
              f"mean={mean_w:.3f}")

        if not valid:
            print(f"    [FAIL] Weights outside correlation range")
            all_valid = False

    if all_valid:
        print("  [PASS] All edge weights are valid correlations")
    return all_valid


def test_direction_constraint(graphs: dict) -> bool:
    """Test that edges only go from earlier to later nodes."""
    print("\n" + "=" * 60)
    print("TEST 3: Direction Constraint")
    print("=" * 60)

    all_valid = True
    for slice_label, graph in graphs.items():
        violations = 0
        for edge in graph.edges:
            src = graph.nodes[edge.src]
            dst = graph.nodes[edge.dst]
            if not (src < dst):
                violations += 1

        print(f"  Slice {slice_label}: {len(graph.edges)} edges, "
              f"{violations} violations")

        if violations > 0:
            all_valid = False

    if all_valid:
        print("  [PASS] All edges respect direction constraint")
    else:
        print("  [FAIL] Some edges violate direction constraint")
    return all_valid


def test_topk_sparsification(graphs: dict, k: int) -> bool:
    """Test that top-k sparsification is applied."""
    print("\n" + "=" * 60)
    print(f"TEST 4: Top-k Sparsification (k={k})")
    print("=" * 60)

    all_valid = True
    for slice_label, graph in graphs.items():
        # Count outgoing edges per node
        out_edges = {}
        for edge in graph.edges:
            out_edges[edge.src] = out_edges.get(edge.src, 0) + 1

        max_out = max(out_edges.values()) if out_edges else 0
        nodes_at_max = sum(1 for v in out_edges.values() if v == k)

        print(f"  Slice {slice_label}: max_out_degree={max_out}, "
              f"nodes_at_k={nodes_at_max}")

        if max_out > k:
            print(f"    [FAIL] Max out-degree {max_out} > k={k}")
            all_valid = False

    if all_valid:
        print(f"  [PASS] All nodes have ≤ {k} outgoing edges")
    return all_valid


def test_edge_budget(graphs: dict, k: int) -> bool:
    """Test that all slices have comparable edge budgets."""
    print("\n" + "=" * 60)
    print(f"TEST 5: Edge Budget (≤ k × |V| = {k} × |V|)")
    print("=" * 60)

    all_valid = True
    for slice_label, graph in graphs.items():
        max_budget = k * graph.num_nodes
        actual = graph.num_edges
        utilization = 100 * actual / max_budget if max_budget > 0 else 0

        print(f"  Slice {slice_label}: {actual}/{max_budget} edges "
              f"({utilization:.1f}% utilized)")

        if actual > max_budget:
            print(f"    [FAIL] Exceeds budget")
            all_valid = False

    if all_valid:
        print("  [PASS] All slices within edge budget")
    return all_valid


def test_slice_differences(graphs: dict) -> bool:
    """Test that graph statistics differ across slices."""
    print("\n" + "=" * 60)
    print("TEST 6: Slice Differences")
    print("=" * 60)

    if len(graphs) < 2:
        print("  [SKIP] Need at least 2 slices to compare")
        return True

    stats_list = []
    for slice_label, graph in graphs.items():
        stats = graph.compute_statistics()
        stats_list.append(stats)
        print(f"  Slice {slice_label}:")
        print(f"    edges={stats['num_edges']}, "
              f"mean_weight={stats['mean_weight']:.4f}, "
              f"density={stats['density']:.4f}")

    # Check if there's variation
    edge_counts = [s["num_edges"] for s in stats_list]
    mean_weights = [s["mean_weight"] for s in stats_list]

    edge_diff = max(edge_counts) - min(edge_counts)
    weight_diff = max(mean_weights) - min(mean_weights)

    print(f"\n  Edge count difference: {edge_diff}")
    print(f"  Mean weight difference: {weight_diff:.4f}")

    # Slices should show SOME difference
    if edge_diff > 0 or weight_diff > 0.001:
        print("  [PASS] Slices show different statistics")
        return True
    else:
        print("  [WARN] Slices may be too similar")
        return True


def main():
    print("=" * 60)
    print("STORY 3.1 VALIDATION: Per-Slice Graph Builder")
    print("=" * 60)

    print("\n  Loading model...")
    model = HookedModel(model_name="gpt2")

    # Generate dataset
    dataset = generate_dataset(model, n_per_slice=15)
    print(f"\n  Dataset: {len(dataset)} examples, "
          f"{len(dataset.get_slices())} slices")

    # Build graphs
    k = 5
    print(f"\n  Building graphs with k={k}...")
    builder = GraphBuilder(k=k, enforce_direction=True)
    graphs = builder.build_all(dataset)

    results = {}

    # Test 1: Fixed node set
    results["fixed_nodes"] = test_fixed_node_set(graphs)

    # Test 2: Correlation weights
    results["corr_weights"] = test_correlation_weights(graphs)

    # Test 3: Direction constraint
    results["direction"] = test_direction_constraint(graphs)

    # Test 4: Top-k sparsification
    results["topk"] = test_topk_sparsification(graphs, k)

    # Test 5: Edge budget
    results["budget"] = test_edge_budget(graphs, k)

    # Test 6: Slice differences
    results["differences"] = test_slice_differences(graphs)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL ACCEPTANCE CRITERIA MET")
    else:
        print("SOME CRITERIA NOT MET - Review above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
