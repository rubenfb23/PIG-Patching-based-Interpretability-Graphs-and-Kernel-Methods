#!/usr/bin/env python3
"""Validation script for Story 1.2 - Paired Prompt Generator.

This script verifies all acceptance criteria:
1. Generator outputs tuples: (x_cln, x_crp, y_star, slice_label, meta)
2. Average observable gap E[O(x^cln) - O(x^crp)] > 0 across generated pairs
3. Generator is reproducible with a fixed random seed
4. Supports at least one task family (MVP: IOI-style indirect object identification)
5. Supports at least one corruption type (MVP: name swap)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pig.model import HookedModel
from pig.prompts import IOIGenerator, NameSwapCorruption, create_ioi_dataset


def test_output_schema() -> bool:
    """Test that generator outputs correct tuple structure."""
    print("\n" + "=" * 60)
    print("TEST 1: Output Schema")
    print("=" * 60)

    gen = IOIGenerator(seed=42)
    pair = gen.generate()
    d = pair.to_dict()

    required_fields = ["x_cln", "x_crp", "y_star", "slice", "meta"]
    slice_fields = ["task", "corruption"]

    print(f"  Generated pair:")
    print(f"    x_cln: '{pair.x_cln}'")
    print(f"    x_crp: '{pair.x_crp}'")
    print(f"    y_star: '{pair.y_star}'")
    print(f"    slice: {pair.slice_label}")
    print(f"    meta: {pair.meta}")

    # Check all required fields
    missing_fields = [f for f in required_fields if f not in d]
    missing_slice = [f for f in slice_fields if f not in d.get("slice", {})]

    if missing_fields:
        print(f"  [FAIL] Missing fields: {missing_fields}")
        return False

    if missing_slice:
        print(f"  [FAIL] Missing slice fields: {missing_slice}")
        return False

    print("  [PASS] All required fields present")
    return True


def test_observable_gap(model: HookedModel, n_examples: int = 50) -> bool:
    """Test that average observable gap is positive."""
    print("\n" + "=" * 60)
    print(f"TEST 2: Observable Gap (n={n_examples})")
    print("=" * 60)

    dataset = create_ioi_dataset(n_examples=n_examples, seed=42)

    gaps = []
    for i, pair in enumerate(dataset):
        clean_score = model.score(pair.x_cln, pair.y_star)
        corrupt_score = model.score(pair.x_crp, pair.y_star)
        gap = clean_score - corrupt_score
        gaps.append(gap)

        if i < 5:
            print(f"  Example {i+1}:")
            print(f"    Clean: '{pair.x_cln[:50]}...' -> {clean_score:.4f}")
            print(f"    Corrupt: '{pair.x_crp[:50]}...' -> {corrupt_score:.4f}")
            print(f"    Gap: {gap:.4f}")

    mean_gap = sum(gaps) / len(gaps)
    positive_gaps = sum(1 for g in gaps if g > 0)

    print(f"\n  Statistics:")
    print(f"    Mean gap E[O(cln) - O(crp)]: {mean_gap:.4f}")
    print(f"    Positive gaps: {positive_gaps}/{len(gaps)} ({100*positive_gaps/len(gaps):.1f}%)")
    print(f"    Min gap: {min(gaps):.4f}")
    print(f"    Max gap: {max(gaps):.4f}")

    # Criterion: mean gap should be positive
    if mean_gap > 0:
        print("  [PASS] Average observable gap is positive")
        return True
    else:
        print("  [FAIL] Average observable gap is not positive")
        return False


def test_reproducibility() -> bool:
    """Test that generator is reproducible with fixed seed."""
    print("\n" + "=" * 60)
    print("TEST 3: Reproducibility")
    print("=" * 60)

    seed = 12345
    n_examples = 20

    # Generate two batches with same seed
    dataset1 = create_ioi_dataset(n_examples=n_examples, seed=seed)
    dataset2 = create_ioi_dataset(n_examples=n_examples, seed=seed)

    matches = 0
    for p1, p2 in zip(dataset1, dataset2):
        if p1.x_cln == p2.x_cln and p1.x_crp == p2.x_crp and p1.y_star == p2.y_star:
            matches += 1

    print(f"  Seed: {seed}")
    print(f"  Examples: {n_examples}")
    print(f"  Matches: {matches}/{n_examples}")

    if matches == n_examples:
        print("  [PASS] Generator is fully reproducible")
        return True
    else:
        print("  [FAIL] Generator is not reproducible")
        return False


def test_task_family() -> bool:
    """Test support for IOI task family."""
    print("\n" + "=" * 60)
    print("TEST 4: Task Family Support (IOI)")
    print("=" * 60)

    gen = IOIGenerator(seed=42)

    print(f"  Task name: {gen.task_name}")
    print(f"  Available templates: {len(gen.TEMPLATES)}")
    print(f"  Available names: {len(gen.NAMES)}")

    # Generate samples and verify IOI structure
    pairs = gen.generate_batch(10)

    valid_count = 0
    for pair in pairs:
        # Verify IOI structure: subject appears first, IO appears second
        # Target (y_star) should be the subject (who gave first)
        name_s = pair.meta["name_s"]
        name_io = pair.meta["name_io"]

        # Check structure
        has_both_names = name_s in pair.x_cln and name_io in pair.x_cln
        target_is_subject = pair.y_star == name_s
        task_is_ioi = pair.slice_label.task == "ioi"

        if has_both_names and target_is_subject and task_is_ioi:
            valid_count += 1

    print(f"  Valid IOI examples: {valid_count}/{len(pairs)}")

    if valid_count == len(pairs):
        print("  [PASS] IOI task family supported correctly")
        return True
    else:
        print("  [FAIL] Some examples have invalid IOI structure")
        return False


def test_corruption_type() -> bool:
    """Test support for name swap corruption."""
    print("\n" + "=" * 60)
    print("TEST 5: Corruption Type Support (name_swap)")
    print("=" * 60)

    corruption = NameSwapCorruption()
    gen = IOIGenerator(corruption=corruption, seed=42)

    print(f"  Corruption type: {corruption.name}")

    pairs = gen.generate_batch(10)

    valid_count = 0
    for pair in pairs:
        name_s = pair.meta["name_s"]

        # In name_swap, subject name should be removed from corrupted prompt
        s_removed = name_s not in pair.x_crp
        corruption_labeled = pair.slice_label.corruption == "name_swap"

        if s_removed and corruption_labeled:
            valid_count += 1

    print(f"  Valid name_swap corruptions: {valid_count}/{len(pairs)}")

    # Show example
    pair = pairs[0]
    print(f"\n  Example:")
    print(f"    Clean: '{pair.x_cln}'")
    print(f"    Corrupt: '{pair.x_crp}'")
    print(f"    Target (S) '{pair.meta['name_s']}' removed from corrupt: "
          f"{'Yes' if pair.meta['name_s'] not in pair.x_crp else 'No'}")

    if valid_count == len(pairs):
        print("  [PASS] Name swap corruption works correctly")
        return True
    else:
        print("  [FAIL] Some corruptions are invalid")
        return False


def main():
    print("=" * 60)
    print("STORY 1.2 VALIDATION: Paired Prompt Generator")
    print("=" * 60)

    results = {}

    # Test 1: Output schema
    results["output_schema"] = test_output_schema()

    # Test 2: Observable gap (requires model)
    print("\n  Loading model for observable gap test...")
    model = HookedModel(model_name="gpt2")
    results["observable_gap"] = test_observable_gap(model, n_examples=50)

    # Test 3: Reproducibility
    results["reproducibility"] = test_reproducibility()

    # Test 4: Task family support
    results["task_family"] = test_task_family()

    # Test 5: Corruption type support
    results["corruption_type"] = test_corruption_type()

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
