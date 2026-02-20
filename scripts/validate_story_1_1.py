#!/usr/bin/env python3
"""Validation script for Story 1.1 - Model Setup and Validation.

This script verifies all acceptance criteria:
1. A small HuggingFace model (e.g., GPT-2) runs locally with frozen weights
2. The system can execute ~10,000 forward passes in a reasonable time
3. Residual activations can be captured at any layer ℓ and token position t
4. Activations can be replaced during a forward pass (hook injection works)
5. Observable delta (Δlogit) changes non-trivially when patching some (ℓ, t) positions
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from pig.model import HookedModel, create_model_from_env


def test_model_setup() -> bool:
    """Test that GPT-2 loads with frozen weights."""
    print("\n" + "=" * 60)
    print("TEST 1: Model Setup with Frozen Weights")
    print("=" * 60)

    model = create_model_from_env()
    wrapped_model = getattr(model, "model", model)

    # Check model is in eval mode
    assert not wrapped_model.training, "Model should be in eval mode"

    # Check all parameters are frozen
    frozen_count = sum(1 for p in wrapped_model.parameters() if not p.requires_grad)
    total_count = sum(1 for _ in wrapped_model.parameters())
    assert frozen_count == total_count, "All parameters should be frozen"

    print(f"  Model: {model.model_name}")
    print(f"  Device: {model.device}")
    print(f"  Layers: {model.n_layers}")
    print(f"  Hidden size: {model.d_model}")
    print(f"  Parameters frozen: {frozen_count}/{total_count}")
    print("  [PASS] Model loaded with frozen weights")

    return True


def test_forward_pass_speed(model: HookedModel, num_passes: int = 1000) -> bool:
    """Test that forward passes are reasonably fast."""
    print("\n" + "=" * 60)
    print(f"TEST 2: Forward Pass Speed ({num_passes} passes)")
    print("=" * 60)

    prompt = "The quick brown fox jumps over the lazy"

    # Warm-up
    for _ in range(10):
        _ = model.score(prompt, "dog")

    # Timed run
    start_time = time.time()
    for _ in range(num_passes):
        _ = model.score(prompt, "dog")
    elapsed = time.time() - start_time

    passes_per_sec = num_passes / elapsed
    time_for_10k = 10000 / passes_per_sec

    print(f"  Completed {num_passes} passes in {elapsed:.2f}s")
    print(f"  Rate: {passes_per_sec:.1f} passes/second")
    print(
        f"  Projected time for 10,000 passes: {time_for_10k:.1f}s ({time_for_10k/60:.1f} min)"
    )
    print(f"  Device: {model.device}")

    # "Reasonable time" depends on device
    # CPU: under 60 minutes for 10k passes (technical notes: CPU support is MVP)
    # GPU: under 10 minutes for 10k passes
    threshold = 600 if model.device == "cuda" else 3600
    reasonable = time_for_10k < threshold
    status = "[PASS]" if reasonable else "[WARN]"
    print(
        f"  {status} {'Reasonable' if reasonable else 'Slow'} for 10k passes on {model.device.upper()}"
    )

    return reasonable


def test_activation_capture(model: HookedModel) -> bool:
    """Test that activations can be captured at any (layer, token) position."""
    print("\n" + "=" * 60)
    print("TEST 3: Activation Capture")
    print("=" * 60)

    prompt = "Hello world"
    cache = model.cache_clean(prompt)

    num_tokens = model.get_num_tokens(prompt)
    num_layers = model.get_num_layers()

    print(f"  Prompt: '{prompt}'")
    print(f"  Tokens: {num_tokens}")
    print(f"  Layers: {num_layers}")

    # Verify we captured activations for all positions
    expected_positions = num_layers * num_tokens
    actual_positions = len(cache)

    print(f"  Cached positions: {actual_positions}/{expected_positions}")

    # Check a few specific positions
    sample_positions = [
        (0, 0),  # First layer, first token
        (num_layers - 1, num_tokens - 1),  # Last layer, last token
        (num_layers // 2, num_tokens // 2),  # Middle
    ]

    for layer, token in sample_positions:
        act = cache.get(layer, token)
        assert act is not None, f"Missing activation at ({layer}, {token})"
        assert act.shape == (model.d_model,), f"Wrong shape at ({layer}, {token})"
        print(
            f"  Position ({layer}, {token}): shape={act.shape}, mean={act.mean():.4f}"
        )

    assert actual_positions == expected_positions, "Missing some positions"
    print("  [PASS] All activations captured correctly")

    return True


def test_activation_patching(model: HookedModel) -> bool:
    """Test that activations can be replaced during forward pass."""
    print("\n" + "=" * 60)
    print("TEST 4: Activation Patching (Hook Injection)")
    print("=" * 60)

    # Use a simple example where patching should change output
    clean_prompt = "The capital of France is"
    corrupted_prompt = "The capital of Germany is"
    target_token = "Paris"

    # Cache clean activations
    clean_cache = model.cache_clean(clean_prompt)

    # Score without patching
    clean_score = model.score(clean_prompt, target_token)
    corrupted_score = model.score(corrupted_prompt, target_token)

    print(f"  Clean prompt: '{clean_prompt}'")
    print(f"  Corrupted prompt: '{corrupted_prompt}'")
    print(f"  Target token: '{target_token}'")
    print(f"  Clean score (O(x_cln)): {clean_score:.4f}")
    print(f"  Corrupted score (O(x_crp)): {corrupted_score:.4f}")

    # Patch a specific position and verify it changes
    num_layers = model.get_num_layers()
    num_tokens = model.get_num_tokens(corrupted_prompt)

    # Patch middle layer, last token (likely contains task-relevant info)
    patch_layer = num_layers // 2
    patch_token = num_tokens - 1

    patched_score = model.patched_score(
        corrupted_prompt,
        target_token,
        clean_cache,
        (patch_layer, patch_token),
    )

    print(f"  Patched position: ({patch_layer}, {patch_token})")
    print(f"  Patched score: {patched_score:.4f}")
    print(f"  Difference from corrupted: {patched_score - corrupted_score:.4f}")

    # Verify patching changes something (even if small)
    different = abs(patched_score - corrupted_score) > 1e-6
    status = "[PASS]" if different else "[WARN]"
    print(f"  {status} Patching {'changes' if different else 'did not change'} output")

    return different


def test_observable_delta(model: HookedModel) -> bool:
    """Test that Δlogit changes non-trivially for some positions."""
    print("\n" + "=" * 60)
    print("TEST 5: Observable Delta (Δlogit)")
    print("=" * 60)

    # IOI-style example
    clean_prompt = "John gave the book to Mary. Mary gave it back to"
    corrupted_prompt = "John gave the book to Mary. John gave it back to"
    target_token = "John"  # Clean should predict John, corrupted predicts Mary

    clean_cache = model.cache_clean(clean_prompt)

    clean_score = model.score(clean_prompt, target_token)
    corrupted_score = model.score(corrupted_prompt, target_token)
    baseline_delta = clean_score - corrupted_score

    print(f"  Clean prompt: '{clean_prompt}'")
    print(f"  Corrupted prompt: '{corrupted_prompt}'")
    print(f"  Target: '{target_token}'")
    print(f"  Clean score: {clean_score:.4f}")
    print(f"  Corrupted score: {corrupted_score:.4f}")
    print(f"  Baseline gap (E[O(cln) - O(crp)]): {baseline_delta:.4f}")

    # Scan for positions with non-trivial patch effects
    num_layers = model.get_num_layers()
    num_tokens = model.get_num_tokens(corrupted_prompt)

    print(f"\n  Scanning {num_layers} layers x {num_tokens} tokens...")

    effects = []
    significant_positions = []

    for layer in range(num_layers):
        for token in range(num_tokens):
            patched_score = model.patched_score(
                corrupted_prompt, target_token, clean_cache, (layer, token)
            )
            effect = patched_score - corrupted_score
            effects.append(effect)
            if abs(effect) > 0.1:  # Non-trivial threshold
                significant_positions.append((layer, token, effect))

    effects_tensor = torch.tensor(effects)
    print(f"\n  Effect statistics:")
    print(f"    Mean: {effects_tensor.mean():.4f}")
    print(f"    Std: {effects_tensor.std():.4f}")
    print(f"    Min: {effects_tensor.min():.4f}")
    print(f"    Max: {effects_tensor.max():.4f}")
    print(f"    Positions with |effect| > 0.1: {len(significant_positions)}")

    if significant_positions:
        print(f"\n  Top significant positions:")
        sorted_positions = sorted(
            significant_positions, key=lambda x: abs(x[2]), reverse=True
        )[:5]
        for layer, token, effect in sorted_positions:
            print(f"    ({layer}, {token}): effect = {effect:.4f}")

    # Success if we find at least some positions with non-trivial effects
    has_signal = len(significant_positions) > 0
    status = "[PASS]" if has_signal else "[WARN]"
    print(
        f"\n  {status} {'Found' if has_signal else 'No'} positions with non-trivial Δlogit"
    )

    return has_signal


def main():
    print("=" * 60)
    print("STORY 1.1 VALIDATION: Model Setup and Validation")
    print("=" * 60)

    results = {}

    # Test 1: Model setup
    results["model_setup"] = test_model_setup()
    model = create_model_from_env()

    # Test 2: Forward pass speed
    results["forward_speed"] = test_forward_pass_speed(model, num_passes=100)

    # Test 3: Activation capture
    results["activation_capture"] = test_activation_capture(model)

    # Test 4: Activation patching
    results["activation_patching"] = test_activation_patching(model)

    # Test 5: Observable delta
    results["observable_delta"] = test_observable_delta(model)

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
        print("SOME CRITERIA NOT MET - Review warnings above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
