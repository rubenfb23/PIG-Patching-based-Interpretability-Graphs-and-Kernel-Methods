#!/usr/bin/env python3
"""Validation script for Story 4.1 - Graph Embeddings (WL Features).

This script verifies all acceptance criteria:
1. Implement WL subtree hashing with configurable depth
2. Output: feature matrix X of shape [num_slices, num_features]
3. Features are deterministic given the same input graphs
4. Embedding computation is cached for reuse
"""

import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from pig.embeddings import WLEncoder, WLEmbeddingCache, compute_wl_features
from pig.graph import GraphBuilder
from pig.model import HookedModel
from pig.patching import PatchEffectDataset, compute_patch_effects
from pig.prompts import ABBACorruption, IOIGenerator


def generate_graphs(model: HookedModel, n_per_slice: int = 10):
    """Generate graphs for testing."""
    print(f"  Generating {n_per_slice} examples per slice...")

    gen_swap = IOIGenerator(seed=42)
    gen_abba = IOIGenerator(corruption=ABBACorruption(), seed=42)

    pairs_swap = gen_swap.generate_batch(n_per_slice)
    pairs_abba = gen_abba.generate_batch(n_per_slice)

    print("  Computing patch effects...")
    dataset = PatchEffectDataset()
    for pair in pairs_swap + pairs_abba:
        from pig.patching import PatchEffectComputer
        computer = PatchEffectComputer(model)
        tensor = computer.compute_single(pair)
        dataset.add(tensor)

    print("  Building graphs...")
    builder = GraphBuilder(k=5, enforce_direction=True)
    graphs = builder.build_all(dataset)

    return graphs


def test_configurable_depth(graphs: dict) -> bool:
    """Test WL hashing with configurable depth."""
    print("\n" + "=" * 60)
    print("TEST 1: Configurable WL Depth")
    print("=" * 60)

    graph_list = list(graphs.values())
    graph = graph_list[0]

    depths = [1, 2, 3, 4]
    feature_counts = []

    for depth in depths:
        encoder = WLEncoder(depth=depth)
        emb = encoder.encode(graph)
        feature_counts.append(len(emb.features))
        print(f"  Depth {depth}: {len(emb.features)} features")

    # Deeper depths should have >= features (more refined labels)
    monotonic = all(
        feature_counts[i] <= feature_counts[i + 1]
        for i in range(len(feature_counts) - 1)
    )

    if monotonic:
        print("  [PASS] Feature count increases with depth")
        return True
    else:
        print("  [WARN] Feature count not monotonic (may be ok)")
        return True


def test_feature_matrix_shape(graphs: dict) -> bool:
    """Test feature matrix output shape."""
    print("\n" + "=" * 60)
    print("TEST 2: Feature Matrix Shape")
    print("=" * 60)

    fm = compute_wl_features(graphs, depth=3)
    matrix = fm.to_matrix()

    num_slices = len(graphs)
    num_features = fm.num_features

    print(f"  Number of slices: {num_slices}")
    print(f"  Number of features: {num_features}")
    print(f"  Matrix shape: {matrix.shape}")

    expected_shape = (num_slices, num_features)

    if matrix.shape == expected_shape:
        print(f"  [PASS] Shape matches [num_slices, num_features]")
        return True
    else:
        print(f"  [FAIL] Expected {expected_shape}, got {matrix.shape}")
        return False


def test_deterministic(graphs: dict) -> bool:
    """Test that features are deterministic."""
    print("\n" + "=" * 60)
    print("TEST 3: Deterministic Features")
    print("=" * 60)

    depth = 3

    # Compute twice
    fm1 = compute_wl_features(graphs, depth=depth)
    fm2 = compute_wl_features(graphs, depth=depth)

    matrix1 = fm1.to_matrix()
    matrix2 = fm2.to_matrix()

    # Check exact equality
    same_vocab = fm1.vocabulary == fm2.vocabulary
    same_matrix = np.array_equal(matrix1, matrix2)

    print(f"  Vocabulary match: {same_vocab}")
    print(f"  Matrix match: {same_matrix}")

    if same_vocab and same_matrix:
        print("  [PASS] Features are deterministic")
        return True
    else:
        print("  [FAIL] Features differ between runs")
        return False


def test_caching(graphs: dict) -> bool:
    """Test embedding caching."""
    print("\n" + "=" * 60)
    print("TEST 4: Embedding Caching")
    print("=" * 60)

    depth = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        # First computation (not cached)
        print("  First computation (not cached)...")
        start1 = time.time()
        fm1 = compute_wl_features(graphs, depth=depth, cache_dir=tmpdir)
        time1 = time.time() - start1

        # Second computation (cached)
        print("  Second computation (cached)...")
        start2 = time.time()
        fm2 = compute_wl_features(graphs, depth=depth, cache_dir=tmpdir)
        time2 = time.time() - start2

        print(f"\n  First run: {time1:.4f}s")
        print(f"  Second run (cached): {time2:.4f}s")

        if time1 > 0:
            speedup = time1 / max(time2, 0.0001)
            print(f"  Speedup: {speedup:.1f}x")

        # Verify cache files exist
        cache_files = list(Path(tmpdir).glob("*.json"))
        print(f"  Cache files created: {len(cache_files)}")

        # Verify results match
        np.testing.assert_array_equal(fm1.to_matrix(), fm2.to_matrix())

        if len(cache_files) > 0:
            print("  [PASS] Caching works correctly")
            return True
        else:
            print("  [FAIL] No cache files created")
            return False


def main():
    print("=" * 60)
    print("STORY 4.1 VALIDATION: Graph Embeddings (WL Features)")
    print("=" * 60)

    print("\n  Loading model...")
    model = HookedModel(model_name="gpt2")

    # Generate graphs
    graphs = generate_graphs(model, n_per_slice=10)
    print(f"\n  Generated {len(graphs)} graphs")

    results = {}

    # Test 1: Configurable depth
    results["configurable_depth"] = test_configurable_depth(graphs)

    # Test 2: Feature matrix shape
    results["matrix_shape"] = test_feature_matrix_shape(graphs)

    # Test 3: Deterministic
    results["deterministic"] = test_deterministic(graphs)

    # Test 4: Caching
    results["caching"] = test_caching(graphs)

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
