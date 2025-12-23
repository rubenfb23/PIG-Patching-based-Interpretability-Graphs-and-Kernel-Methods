"""Paired prompt generation for patching experiments.

This module provides generators for clean/corrupted prompt pairs used in
activation patching experiments. Each generator produces controlled inputs
where the clean prompt succeeds and the corrupted prompt fails at a target behavior.

MVP implements the IOI (Indirect Object Identification) task with name-swap corruption.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class SliceLabel:
    """Identifies a slice (task + corruption combination)."""

    task: str
    corruption: str

    def __str__(self) -> str:
        return f"{self.task}:{self.corruption}"


@dataclass(frozen=True)
class PromptPair:
    """A clean/corrupted prompt pair for patching experiments.

    Attributes:
        x_cln: The clean prompt where the model should predict y_star
        x_crp: The corrupted prompt where the model fails to predict y_star
        y_star: The target token that the clean prompt should predict
        slice_label: Identifies the task and corruption type
        meta: Additional metadata (names, template info, etc.)
    """

    x_cln: str
    x_crp: str
    y_star: str
    slice_label: SliceLabel
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary format matching the schema."""
        return {
            "x_cln": self.x_cln,
            "x_crp": self.x_crp,
            "y_star": self.y_star,
            "slice": {
                "task": self.slice_label.task,
                "corruption": self.slice_label.corruption,
            },
            "meta": self.meta,
        }


@runtime_checkable
class CorruptionStrategy(Protocol):
    """Protocol for prompt corruption strategies."""

    @property
    def name(self) -> str:
        """Return the name of this corruption strategy."""
        ...

    def corrupt(self, clean_prompt: str, meta: dict) -> str:
        """Apply corruption to a clean prompt.

        Args:
            clean_prompt: The original clean prompt
            meta: Metadata containing information needed for corruption

        Returns:
            The corrupted prompt
        """
        ...


class NameSwapCorruption:
    """Corruption strategy that swaps the subject name with the IO name.

    For IOI task: replaces the subject name (S) with the indirect object name (IO).
    This reduces the probability of predicting S as the next token since S no longer
    appears in the corrupted prompt.

    Example:
        Clean: "John gave the book to Mary. Mary gave it back to" -> predict "John"
        Corrupt: "Mary gave the book to Mary. Mary gave it back to" -> predict "Mary"

    The target (y_star) is the subject name (S), which should have higher logit
    in the clean prompt than in the corrupted prompt where it doesn't appear.
    """

    @property
    def name(self) -> str:
        return "name_swap"

    def corrupt(self, clean_prompt: str, meta: dict) -> str:
        """Replace the subject name with the IO name in the prompt.

        Args:
            clean_prompt: The clean prompt containing both names
            meta: Must contain 'name_s' (subject) and 'name_io' (indirect object)

        Returns:
            Prompt with name_s replaced by name_io
        """
        name_s = meta["name_s"]
        name_io = meta["name_io"]
        # Replace the subject name with the indirect object name
        # This makes the target (S) absent from the corrupted prompt
        return clean_prompt.replace(name_s, name_io)


class PromptGenerator(ABC):
    """Abstract base class for prompt pair generators."""

    def __init__(self, seed: int | None = None):
        """Initialize the generator with an optional random seed.

        Args:
            seed: Random seed for reproducibility. If None, uses system randomness.
        """
        self._seed = seed
        self._rng = random.Random(seed)

    @property
    def seed(self) -> int | None:
        """Return the random seed used by this generator."""
        return self._seed

    def reset(self, seed: int | None = None) -> None:
        """Reset the generator with a new seed.

        Args:
            seed: New seed. If None, uses the original seed.
        """
        self._seed = seed if seed is not None else self._seed
        self._rng = random.Random(self._seed)

    @abstractmethod
    def generate(self) -> PromptPair:
        """Generate a single prompt pair.

        Returns:
            A PromptPair with clean and corrupted prompts
        """
        ...

    def generate_batch(self, n: int) -> list[PromptPair]:
        """Generate a batch of prompt pairs.

        Args:
            n: Number of pairs to generate

        Returns:
            List of n PromptPair objects
        """
        return [self.generate() for _ in range(n)]

    def __iter__(self) -> Iterator[PromptPair]:
        """Infinite iterator over generated prompt pairs."""
        while True:
            yield self.generate()


class IOIGenerator(PromptGenerator):
    """Generator for Indirect Object Identification (IOI) task.

    The IOI task tests whether a model can identify the indirect object
    in sentences like "X gave something to Y. Y gave it back to ___"
    where the answer should be X.

    Corruption strategies modify the prompt to make this identification harder.
    """

    # Common English names for variety
    NAMES: list[str] = [
        "John", "Mary", "Alice", "Bob", "Emma", "James",
        "Sarah", "Michael", "Lisa", "David", "Jennifer", "Robert",
        "Emily", "William", "Jessica", "Thomas", "Ashley", "Daniel",
        "Amanda", "Matthew", "Nicole", "Christopher", "Stephanie", "Andrew",
    ]

    # Template patterns for IOI sentences
    # Format: (template, subject_position, io_position)
    # Template uses {S} for subject (who gave first) and {IO} for indirect object
    TEMPLATES: list[tuple[str, str]] = [
        ("{S} gave the book to {IO}. {IO} gave it back to", "gave the book"),
        ("{S} lent the money to {IO}. {IO} returned it to", "lent the money"),
        ("{S} passed the ball to {IO}. {IO} threw it to", "passed the ball"),
        ("{S} sent a letter to {IO}. {IO} replied to", "sent a letter"),
        ("{S} handed the keys to {IO}. {IO} returned them to", "handed the keys"),
        ("{S} showed the photo to {IO}. {IO} gave it back to", "showed the photo"),
    ]

    def __init__(
        self,
        corruption: CorruptionStrategy | None = None,
        seed: int | None = None,
        names: list[str] | None = None,
        templates: list[tuple[str, str]] | None = None,
    ):
        """Initialize the IOI generator.

        Args:
            corruption: Corruption strategy to use (default: NameSwapCorruption)
            seed: Random seed for reproducibility
            names: Custom list of names to use (default: built-in NAMES)
            templates: Custom templates (default: built-in TEMPLATES)
        """
        super().__init__(seed)
        self._corruption = corruption or NameSwapCorruption()
        self._names = names or self.NAMES
        self._templates = templates or self.TEMPLATES

    @property
    def task_name(self) -> str:
        """Return the task name."""
        return "ioi"

    def generate(self) -> PromptPair:
        """Generate an IOI prompt pair.

        Returns:
            PromptPair with:
            - x_cln: Clean prompt where model should predict the subject name
            - x_crp: Corrupted prompt (name-swapped)
            - y_star: The subject name (correct answer)
        """
        # Select two different names
        name_s, name_io = self._rng.sample(self._names, 2)

        # Select a template
        template, action = self._rng.choice(self._templates)

        # Build clean prompt
        x_cln = template.format(S=name_s, IO=name_io)

        # Store metadata
        meta = {
            "name_s": name_s,
            "name_io": name_io,
            "template": template,
            "action": action,
        }

        # Apply corruption
        x_crp = self._corruption.corrupt(x_cln, meta)

        # Target is the subject name (who gave first, should receive back)
        y_star = name_s

        slice_label = SliceLabel(task=self.task_name, corruption=self._corruption.name)

        return PromptPair(
            x_cln=x_cln,
            x_crp=x_crp,
            y_star=y_star,
            slice_label=slice_label,
            meta=meta,
        )


class ABBACorruption:
    """ABBA corruption: Swap positions to create ABBA pattern from ABAB.

    Original: "A gave to B. B gave back to" (answer: A)
    Corrupted: "B gave to A. A gave back to" (answer: B, but uses same names)
    """

    @property
    def name(self) -> str:
        return "abba"

    def corrupt(self, clean_prompt: str, meta: dict) -> str:
        """Swap the positions of S and IO in the template."""
        name_s = meta["name_s"]
        name_io = meta["name_io"]
        template = meta["template"]
        # Swap the roles: S becomes IO and IO becomes S
        return template.format(S=name_io, IO=name_s)


def create_ioi_dataset(
    n_examples: int,
    corruption: str = "name_swap",
    seed: int = 42,
) -> list[PromptPair]:
    """Convenience function to create an IOI dataset.

    Args:
        n_examples: Number of examples to generate
        corruption: Corruption type ("name_swap" or "abba")
        seed: Random seed for reproducibility

    Returns:
        List of PromptPair objects
    """
    corruption_map: dict[str, CorruptionStrategy] = {
        "name_swap": NameSwapCorruption(),
        "abba": ABBACorruption(),
    }

    if corruption not in corruption_map:
        raise ValueError(f"Unknown corruption: {corruption}. Available: {list(corruption_map.keys())}")

    generator = IOIGenerator(corruption=corruption_map[corruption], seed=seed)
    return generator.generate_batch(n_examples)
