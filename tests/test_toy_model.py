"""Unit tests for the tiny hand-made transformer test model."""

import numpy as np
import pytest

from pig.patching import compute_patch_effects
from pig.prompts import create_ioi_dataset
from pig.toy_model import TinyTransformerConfig, ToyHookedModel


def _make_model() -> ToyHookedModel:
    return ToyHookedModel(
        config=TinyTransformerConfig(
            vocab_size=257,
            max_seq_len=96,
            d_model=32,
            n_layers=2,
            n_heads=4,
            mlp_dim=64,
            seed=7,
        ),
        device="cpu",
    )


def test_basic_scoring_api():
    model = _make_model()
    score = model.score("John gave the book to Mary .", "John")
    assert isinstance(score, float)
    assert not np.isnan(score)


def test_cache_and_patch_single_node():
    model = _make_model()

    clean = "John gave the book to Mary . Mary gave it back to"
    corrupt = "Mary gave the book to Mary . Mary gave it back to"
    target = "John"

    cache = model.cache_clean(clean)
    base = model.score(corrupt, target)
    patched = model.patched_score(corrupt, target, cache, (0, 0))

    assert isinstance(patched, float)
    assert not np.isnan(patched)
    assert patched != pytest.approx(base)


def test_cache_components_and_attention_patch():
    model = _make_model()
    clean = "Alice sent a letter to Bob . Bob replied to"
    corrupt = "Bob sent a letter to Bob . Bob replied to"

    cache = model.cache_clean_components(clean, node_types=("res", "mlp", "att"))
    # First layer, first token, first attention head should be cached.
    assert cache.get(0, 0, node_type="att", head=0) is not None
    assert cache.get(0, 0, node_type="mlp") is not None
    assert cache.get(0, 0, node_type="res") is not None

    patched = model.patched_score(corrupt, "Alice", cache, (0, 0, "att", 0))
    assert isinstance(patched, float)


def test_integration_with_patch_effect_pipeline():
    model = _make_model()
    pairs = create_ioi_dataset(n_examples=2, corruption="name_swap", seed=11)

    dataset = compute_patch_effects(
        model,
        pairs,
        show_progress=False,
        node_types=("res", "mlp", "att"),
    )

    assert len(dataset) == 2
    tensor = dataset[0]
    assert tensor.num_layers == model.n_layers
    assert tensor.num_tokens > 0
    assert tensor.num_components == model.n_heads + 2  # att heads + mlp + res


def test_invalid_attention_patch_node_requires_head():
    model = _make_model()
    pair = create_ioi_dataset(n_examples=1, corruption="name_swap", seed=3)[0]
    cache = model.cache_clean_components(pair.x_cln, node_types=("att",))

    with pytest.raises(ValueError):
        model.patched_score(pair.x_crp, pair.y_star, cache, (0, 0, "att"))
