"""Unit tests for pig.patching module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pig.patching import (
    PatchEffectCache,
    PatchEffectDataset,
    PatchEffectTensor,
)
from pig.prompts import PromptPair, SliceLabel


class TestPatchEffectTensor:
    """Tests for PatchEffectTensor class."""

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        effects = np.random.randn(12, 8).astype(np.float32)
        effects[5, 3] = 5.0  # Significant positive
        effects[7, 2] = -3.0  # Significant negative

        pair = PromptPair(
            x_cln="John gave to Mary",
            x_crp="Mary gave to Mary",
            y_star="John",
            slice_label=SliceLabel(task="ioi", corruption="name_swap"),
            meta={"name_s": "John", "name_io": "Mary"},
        )

        return PatchEffectTensor(
            effects=effects,
            prompt_pair=pair,
            base_score=-80.0,
            clean_score=-70.0,
        )

    def test_properties(self, sample_tensor):
        """Test tensor properties."""
        assert sample_tensor.num_layers == 12
        assert sample_tensor.num_tokens == 8
        assert sample_tensor.shape == (12, 8)

    def test_get_effect(self, sample_tensor):
        """Test getting effect at specific position."""
        effect = sample_tensor.get_effect(5, 3)
        assert effect == pytest.approx(5.0)

    def test_get_significant_positions(self, sample_tensor):
        """Test finding significant positions."""
        positions = sample_tensor.get_significant_positions(threshold=2.0)

        # Should find at least our two planted significant positions
        assert len(positions) >= 2

        # Check sorting by |effect|
        for i in range(len(positions) - 1):
            assert abs(positions[i][2]) >= abs(positions[i + 1][2])

    def test_to_dict_from_dict(self, sample_tensor):
        """Test serialization roundtrip."""
        data = sample_tensor.to_dict()

        assert "effects" in data
        assert "prompt_pair" in data
        assert "base_score" in data
        assert "clean_score" in data

        restored = PatchEffectTensor.from_dict(data)

        np.testing.assert_array_almost_equal(
            restored.effects, sample_tensor.effects
        )
        assert restored.base_score == sample_tensor.base_score
        assert restored.clean_score == sample_tensor.clean_score
        assert restored.prompt_pair.x_cln == sample_tensor.prompt_pair.x_cln


class TestPatchEffectDataset:
    """Tests for PatchEffectDataset class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset with multiple tensors."""
        dataset = PatchEffectDataset()

        slices = [
            SliceLabel(task="ioi", corruption="name_swap"),
            SliceLabel(task="ioi", corruption="abba"),
        ]

        for i in range(10):
            slice_label = slices[i % 2]
            pair = PromptPair(
                x_cln=f"Clean {i}",
                x_crp=f"Corrupt {i}",
                y_star="target",
                slice_label=slice_label,
                meta={},
            )
            tensor = PatchEffectTensor(
                effects=np.random.randn(12, 8).astype(np.float32),
                prompt_pair=pair,
                base_score=-80.0,
                clean_score=-70.0,
            )
            dataset.add(tensor)

        return dataset

    def test_len(self, sample_dataset):
        """Test dataset length."""
        assert len(sample_dataset) == 10

    def test_iteration(self, sample_dataset):
        """Test iteration over dataset."""
        count = 0
        for tensor in sample_dataset:
            assert isinstance(tensor, PatchEffectTensor)
            count += 1
        assert count == 10

    def test_indexing(self, sample_dataset):
        """Test indexing."""
        tensor = sample_dataset[0]
        assert isinstance(tensor, PatchEffectTensor)

    def test_get_slices(self, sample_dataset):
        """Test getting unique slices."""
        slices = sample_dataset.get_slices()
        assert len(slices) == 2

    def test_get_by_slice(self, sample_dataset):
        """Test filtering by slice."""
        slice_label = SliceLabel(task="ioi", corruption="name_swap")
        tensors = sample_dataset.get_by_slice(slice_label)
        assert len(tensors) == 5

    def test_get_effect_matrix(self, sample_dataset):
        """Test getting stacked effect matrix."""
        slice_label = SliceLabel(task="ioi", corruption="name_swap")
        matrix = sample_dataset.get_effect_matrix(slice_label)

        assert matrix.shape == (5, 12 * 8)  # 5 examples, 96 positions

    def test_compute_statistics(self, sample_dataset):
        """Test computing statistics."""
        stats = sample_dataset.compute_statistics()

        assert stats["num_examples"] == 10
        assert stats["num_slices"] == 2
        assert "effect_mean" in stats
        assert "effect_std" in stats


class TestPatchEffectCache:
    """Tests for PatchEffectCache class."""

    @pytest.fixture
    def temp_cache(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield PatchEffectCache(tmpdir)

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        pair = PromptPair(
            x_cln="John gave to Mary",
            x_crp="Mary gave to Mary",
            y_star="John",
            slice_label=SliceLabel(task="ioi", corruption="name_swap"),
            meta={"name_s": "John", "name_io": "Mary"},
        )
        return PatchEffectTensor(
            effects=np.random.randn(12, 8).astype(np.float32),
            prompt_pair=pair,
            base_score=-80.0,
            clean_score=-70.0,
        )

    def test_put_and_get(self, temp_cache, sample_tensor):
        """Test storing and retrieving from cache."""
        model_name = "gpt2"

        # Initially not in cache
        result = temp_cache.get(sample_tensor.prompt_pair, model_name)
        assert result is None

        # Store
        temp_cache.put(sample_tensor, model_name)

        # Now should be retrievable
        result = temp_cache.get(sample_tensor.prompt_pair, model_name)
        assert result is not None
        np.testing.assert_array_almost_equal(
            result.effects, sample_tensor.effects
        )

    def test_clear(self, temp_cache, sample_tensor):
        """Test clearing the cache."""
        model_name = "gpt2"

        temp_cache.put(sample_tensor, model_name)
        assert temp_cache.get(sample_tensor.prompt_pair, model_name) is not None

        count = temp_cache.clear()
        assert count == 1

        assert temp_cache.get(sample_tensor.prompt_pair, model_name) is None
