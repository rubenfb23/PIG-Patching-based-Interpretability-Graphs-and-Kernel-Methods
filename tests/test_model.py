"""Unit tests for pig.model module."""

import pytest
import torch

from pig.model import ActivationCache, HookedModel


class TestActivationCache:
    """Tests for ActivationCache class."""

    def test_store_and_get(self):
        """Test storing and retrieving activations."""
        cache = ActivationCache()
        activation = torch.randn(768)

        cache.store(0, 0, activation)

        retrieved = cache.get(0, 0)
        assert retrieved is not None
        assert torch.allclose(retrieved, activation)

    def test_get_missing(self):
        """Test retrieving non-existent activation returns None."""
        cache = ActivationCache()
        assert cache.get(0, 0) is None

    def test_store_creates_clone(self):
        """Test that stored activations are clones (not references)."""
        cache = ActivationCache()
        activation = torch.randn(768)
        original_sum = activation.sum().item()

        cache.store(0, 0, activation)

        # Modify original
        activation.fill_(0)

        # Cached should be unchanged
        retrieved = cache.get(0, 0)
        assert retrieved.sum().item() == pytest.approx(original_sum)

    def test_get_all_for_layer(self):
        """Test getting all activations for a specific layer."""
        cache = ActivationCache()

        cache.store(0, 0, torch.randn(768))
        cache.store(0, 1, torch.randn(768))
        cache.store(1, 0, torch.randn(768))

        layer_0_acts = cache.get_all_for_layer(0)
        assert len(layer_0_acts) == 2
        assert 0 in layer_0_acts
        assert 1 in layer_0_acts

    def test_clear(self):
        """Test clearing the cache."""
        cache = ActivationCache()
        cache.store(0, 0, torch.randn(768))
        assert len(cache) == 1

        cache.clear()
        assert len(cache) == 0

    def test_len(self):
        """Test cache length."""
        cache = ActivationCache()
        assert len(cache) == 0

        cache.store(0, 0, torch.randn(768))
        cache.store(0, 1, torch.randn(768))
        assert len(cache) == 2


@pytest.mark.slow
class TestHookedModel:
    """Integration tests for HookedModel (requires model download)."""

    @pytest.fixture(scope="class")
    def model(self):
        """Create a model instance for tests."""
        return HookedModel(model_name="gpt2", device="cpu")

    def test_model_frozen(self, model):
        """Test that model weights are frozen."""
        for param in model.model.parameters():
            assert not param.requires_grad

    def test_model_eval_mode(self, model):
        """Test that model is in eval mode."""
        assert not model.model.training

    def test_tokenize(self, model):
        """Test tokenization."""
        tokens = model.tokenize("Hello world")
        assert tokens.shape[0] == 1  # batch size
        assert tokens.shape[1] == 2  # two tokens

    def test_get_token_id(self, model):
        """Test getting token ID."""
        token_id = model.get_token_id("Paris")
        assert isinstance(token_id, int)
        assert 0 <= token_id < model.vocab_size

    def test_score(self, model):
        """Test computing observable (logit score)."""
        score = model.score("The capital of France is", "Paris")
        assert isinstance(score, float)
        assert not torch.isnan(torch.tensor(score))

    def test_cache_clean(self, model):
        """Test caching clean activations."""
        cache = model.cache_clean("Hello world")

        num_tokens = model.get_num_tokens("Hello world")
        num_layers = model.get_num_layers()

        assert len(cache) == num_tokens * num_layers

        # Check specific positions
        act = cache.get(0, 0)
        assert act is not None
        assert act.shape == (model.d_model,)

    def test_patched_score_changes_output(self, model):
        """Test that patching changes the model output."""
        clean_prompt = "The capital of France is"
        corrupted_prompt = "The capital of Germany is"
        target = "Paris"

        cache = model.cache_clean(clean_prompt)

        base_score = model.score(corrupted_prompt, target)
        patched_score = model.patched_score(
            corrupted_prompt, target, cache, (6, 4)
        )

        # Patching should change the output (not necessarily improve it)
        assert base_score != patched_score

    def test_get_num_layers(self, model):
        """Test getting number of layers."""
        assert model.get_num_layers() == 12  # GPT-2 has 12 layers

    def test_get_num_tokens(self, model):
        """Test getting number of tokens."""
        assert model.get_num_tokens("Hello world") == 2
        assert model.get_num_tokens("The quick brown fox") == 4
