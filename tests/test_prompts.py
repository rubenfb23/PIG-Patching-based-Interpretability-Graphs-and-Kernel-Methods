"""Unit tests for pig.prompts module."""

import pytest

from pig.prompts import (
    ABBACorruption,
    IOIGenerator,
    NameSwapCorruption,
    PromptPair,
    SecondSubjectSwapCorruption,
    SliceLabel,
    create_ioi_dataset,
    get_prompt_pair_distractor,
)


class TestSliceLabel:
    """Tests for SliceLabel class."""

    def test_creation(self):
        """Test SliceLabel creation."""
        label = SliceLabel(task="ioi", corruption="name_swap")
        assert label.task == "ioi"
        assert label.corruption == "name_swap"

    def test_str_representation(self):
        """Test string representation."""
        label = SliceLabel(task="ioi", corruption="name_swap")
        assert str(label) == "ioi:name_swap"

    def test_immutability(self):
        """Test that SliceLabel is frozen/immutable."""
        label = SliceLabel(task="ioi", corruption="name_swap")
        with pytest.raises(AttributeError):
            label.task = "other"  # type: ignore


class TestPromptPair:
    """Tests for PromptPair class."""

    def test_creation(self):
        """Test PromptPair creation."""
        label = SliceLabel(task="ioi", corruption="name_swap")
        pair = PromptPair(
            x_cln="John gave to Mary",
            x_crp="John gave to John",
            y_star="John",
            slice_label=label,
            meta={"name_s": "John", "name_io": "Mary"},
        )
        assert pair.x_cln == "John gave to Mary"
        assert pair.x_crp == "John gave to John"
        assert pair.y_star == "John"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        label = SliceLabel(task="ioi", corruption="name_swap")
        pair = PromptPair(
            x_cln="John gave to Mary",
            x_crp="John gave to John",
            y_star="John",
            slice_label=label,
            meta={"name_s": "John"},
        )
        d = pair.to_dict()
        assert d["x_cln"] == "John gave to Mary"
        assert d["x_crp"] == "John gave to John"
        assert d["y_star"] == "John"
        assert d["slice"]["task"] == "ioi"
        assert d["slice"]["corruption"] == "name_swap"
        assert d["meta"]["name_s"] == "John"


class TestNameSwapCorruption:
    """Tests for NameSwapCorruption strategy."""

    def test_name_property(self):
        """Test corruption name."""
        corruption = NameSwapCorruption()
        assert corruption.name == "name_swap"

    def test_corrupt(self):
        """Test name swap corruption (S replaced with IO)."""
        corruption = NameSwapCorruption()
        meta = {"name_s": "John", "name_io": "Mary"}
        clean = "John gave the book to Mary. Mary gave it back to"
        corrupted = corruption.corrupt(clean, meta)
        # Subject "John" is replaced with IO "Mary"
        assert corrupted == "Mary gave the book to Mary. Mary gave it back to"

    def test_multiple_occurrences(self):
        """Test that all occurrences of subject are replaced."""
        corruption = NameSwapCorruption()
        meta = {"name_s": "Bob", "name_io": "Alice"}
        clean = "Bob met Alice. Alice said hello to Bob. Alice smiled at"
        corrupted = corruption.corrupt(clean, meta)
        # Subject "Bob" should be completely removed
        assert "Bob" not in corrupted
        # Original: 2 Bob + 3 Alice -> After swap: 5 Alice
        assert corrupted.count("Alice") == 5


class TestABBACorruption:
    """Tests for ABBACorruption strategy."""

    def test_name_property(self):
        """Test corruption name."""
        corruption = ABBACorruption()
        assert corruption.name == "abba"

    def test_corrupt(self):
        """Test ABBA corruption (role swap)."""
        corruption = ABBACorruption()
        meta = {
            "name_s": "John",
            "name_io": "Mary",
            "template": "{S} gave the book to {IO}. {IO} gave it back to",
        }
        clean = "John gave the book to Mary. Mary gave it back to"
        corrupted = corruption.corrupt(clean, meta)
        # After swap: Mary is now S, John is now IO
        assert corrupted == "Mary gave the book to John. John gave it back to"


class TestSecondSubjectSwapCorruption:
    """Tests for the surface-balanced second-subject swap corruption."""

    def test_name_property(self):
        corruption = SecondSubjectSwapCorruption()
        assert corruption.name == "second_subject_swap"

    def test_corrupt(self):
        corruption = SecondSubjectSwapCorruption()
        meta = {
            "name_s": "John",
            "name_io": "Mary",
            "template": "{S} gave the book to {IO}. {IO} gave it back to",
        }
        clean = "John gave the book to Mary. Mary gave it back to"
        corrupted = corruption.corrupt(clean, meta)

        assert corrupted == "John gave the book to Mary. John gave it back to"
        assert corrupted.count("John") == 2
        assert corrupted.count("Mary") == 1

    def test_name_counts_match_abba(self):
        meta = {
            "name_s": "John",
            "name_io": "Mary",
            "template": "{S} gave the book to {IO}. {IO} gave it back to",
        }
        clean = "John gave the book to Mary. Mary gave it back to"

        abba = ABBACorruption().corrupt(clean, meta)
        balanced = SecondSubjectSwapCorruption().corrupt(clean, meta)

        assert abba.count("John") == balanced.count("John") == 2
        assert abba.count("Mary") == balanced.count("Mary") == 1


class TestIOIGenerator:
    """Tests for IOIGenerator class."""

    def test_generate_returns_prompt_pair(self):
        """Test that generate returns a PromptPair."""
        gen = IOIGenerator(seed=42)
        pair = gen.generate()
        assert isinstance(pair, PromptPair)

    def test_generate_has_required_fields(self):
        """Test that generated pairs have all required fields."""
        gen = IOIGenerator(seed=42)
        pair = gen.generate()

        assert pair.x_cln  # Non-empty clean prompt
        assert pair.x_crp  # Non-empty corrupted prompt
        assert pair.y_star  # Non-empty target token
        assert pair.slice_label.task == "ioi"
        assert pair.slice_label.corruption == "name_swap"

    def test_clean_contains_both_names(self):
        """Test that clean prompt contains both subject and IO names."""
        gen = IOIGenerator(seed=42)
        pair = gen.generate()

        name_s = pair.meta["name_s"]
        name_io = pair.meta["name_io"]

        assert name_s in pair.x_cln
        assert name_io in pair.x_cln
        assert name_s != name_io

    def test_corrupted_removes_subject_name(self):
        """Test that corrupted prompt has subject replaced with IO."""
        gen = IOIGenerator(seed=42)
        pair = gen.generate()

        name_s = pair.meta["name_s"]
        # Subject name should be completely removed from corrupted prompt
        assert name_s not in pair.x_crp

    def test_target_is_subject(self):
        """Test that target token is the subject name."""
        gen = IOIGenerator(seed=42)
        pair = gen.generate()
        assert pair.y_star == pair.meta["name_s"]

    def test_generated_pair_exposes_distractor(self):
        """Test that IOI pairs carry the opposite-name distractor."""
        gen = IOIGenerator(seed=42)
        pair = gen.generate()

        assert pair.meta["y_distractor"] == pair.meta["name_io"]
        assert get_prompt_pair_distractor(pair) == pair.meta["name_io"]

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        gen1 = IOIGenerator(seed=123)
        gen2 = IOIGenerator(seed=123)

        pairs1 = gen1.generate_batch(10)
        pairs2 = gen2.generate_batch(10)

        for p1, p2 in zip(pairs1, pairs2):
            assert p1.x_cln == p2.x_cln
            assert p1.x_crp == p2.x_crp
            assert p1.y_star == p2.y_star

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        gen1 = IOIGenerator(seed=1)
        gen2 = IOIGenerator(seed=2)

        pairs1 = gen1.generate_batch(100)
        pairs2 = gen2.generate_batch(100)

        # At least some pairs should be different
        differences = sum(
            1 for p1, p2 in zip(pairs1, pairs2) if p1.x_cln != p2.x_cln
        )
        assert differences > 0

    def test_generate_batch(self):
        """Test batch generation."""
        gen = IOIGenerator(seed=42)
        pairs = gen.generate_batch(50)

        assert len(pairs) == 50
        assert all(isinstance(p, PromptPair) for p in pairs)

    def test_iterator(self):
        """Test that generator is iterable."""
        gen = IOIGenerator(seed=42)
        pairs = []
        for i, pair in enumerate(gen):
            pairs.append(pair)
            if i >= 9:
                break

        assert len(pairs) == 10
        assert all(isinstance(p, PromptPair) for p in pairs)

    def test_reset(self):
        """Test resetting the generator."""
        gen = IOIGenerator(seed=42)
        first_batch = gen.generate_batch(5)

        gen.reset()
        second_batch = gen.generate_batch(5)

        for p1, p2 in zip(first_batch, second_batch):
            assert p1.x_cln == p2.x_cln

    def test_custom_names(self):
        """Test using custom names."""
        custom_names = ["Alpha", "Beta", "Gamma"]
        gen = IOIGenerator(seed=42, names=custom_names)
        pairs = gen.generate_batch(20)

        all_names_used = set()
        for pair in pairs:
            all_names_used.add(pair.meta["name_s"])
            all_names_used.add(pair.meta["name_io"])

        assert all_names_used.issubset(set(custom_names))

    def test_abba_corruption(self):
        """Test with ABBA corruption strategy."""
        gen = IOIGenerator(corruption=ABBACorruption(), seed=42)
        pair = gen.generate()

        assert pair.slice_label.corruption == "abba"

    def test_second_subject_swap_corruption(self):
        """Test with surface-balanced second-subject swap corruption."""
        gen = IOIGenerator(corruption=SecondSubjectSwapCorruption(), seed=42)
        pair = gen.generate()

        assert pair.slice_label.corruption == "second_subject_swap"


class TestCreateIOIDataset:
    """Tests for create_ioi_dataset convenience function."""

    def test_creates_correct_number(self):
        """Test that correct number of examples are created."""
        dataset = create_ioi_dataset(n_examples=25, seed=42)
        assert len(dataset) == 25

    def test_name_swap_corruption(self):
        """Test with name_swap corruption."""
        dataset = create_ioi_dataset(
            n_examples=10, corruption="name_swap", seed=42
        )
        assert all(p.slice_label.corruption == "name_swap" for p in dataset)

    def test_abba_corruption(self):
        """Test with abba corruption."""
        dataset = create_ioi_dataset(n_examples=10, corruption="abba", seed=42)
        assert all(p.slice_label.corruption == "abba" for p in dataset)

    def test_second_subject_swap_corruption(self):
        """Test with second_subject_swap corruption."""
        dataset = create_ioi_dataset(
            n_examples=10,
            corruption="second_subject_swap",
            seed=42,
        )
        assert all(
            p.slice_label.corruption == "second_subject_swap" for p in dataset
        )

    def test_invalid_corruption_raises(self):
        """Test that invalid corruption type raises error."""
        with pytest.raises(ValueError):
            create_ioi_dataset(n_examples=10, corruption="invalid")

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        d1 = create_ioi_dataset(n_examples=20, seed=99)
        d2 = create_ioi_dataset(n_examples=20, seed=99)

        for p1, p2 in zip(d1, d2):
            assert p1.x_cln == p2.x_cln
            assert p1.y_star == p2.y_star

    def test_missing_or_invalid_distractor_returns_none(self):
        """Test distractor helper for legacy prompt pairs."""
        pair = PromptPair(
            x_cln="John gave to Mary",
            x_crp="Mary gave to Mary",
            y_star="John",
            slice_label=SliceLabel(task="ioi", corruption="name_swap"),
            meta={},
        )
        assert get_prompt_pair_distractor(pair) is None
