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
from typing import Iterator, Protocol, Sequence, runtime_checkable


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


def get_prompt_pair_distractor(pair: PromptPair) -> str | None:
    """Return the contrast token for pair-level observables when available.

    IOI-style experiments are more robust when the observable is a logit margin
    between the correct name and the distractor name instead of the raw target
    logit alone. We keep this optional to preserve compatibility with older
    prompt pairs and non-IOI tasks.
    """
    value = pair.meta.get("y_distractor")
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or normalized == pair.y_star:
        return None
    return normalized


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
        # Filter to names that are typically single tokens in GPT-2
        single_token_names = [
            "John", "Mary", "Alice", "Bob", "James",
            "Lisa", "David", "Emily", "Thomas", "Daniel",
        ]
        self._names = names or single_token_names
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
            "y_distractor": name_io,
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


class SecondSubjectSwapCorruption:
    """Surface-balanced IOI corruption with the same name counts as ABBA.

    Clean:     S gave the object to IO. IO gave it back to ___  (answer: S)
    Corrupt:   S gave the object to IO. S  gave it back to ___

    The corrupted prompt contains the target name twice and the distractor name
    once, matching the ABBA corruption's target/distractor counts while changing
    a different role position.
    """

    @property
    def name(self) -> str:
        return "second_subject_swap"

    def corrupt(self, clean_prompt: str, meta: dict) -> str:
        name_s = meta["name_s"]
        name_io = meta["name_io"]
        template = meta["template"]
        first_sentence, second_sentence = template.split(". ", maxsplit=1)
        first = first_sentence.format(S=name_s, IO=name_io)
        second = second_sentence.format(S=name_s, IO=name_s)
        return f"{first}. {second}"


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
        "second_subject_swap": SecondSubjectSwapCorruption(),
    }

    if corruption not in corruption_map:
        raise ValueError(f"Unknown corruption: {corruption}. Available: {list(corruption_map.keys())}")

    generator = IOIGenerator(corruption=corruption_map[corruption], seed=seed)
    return generator.generate_batch(n_examples)


# ------ GreaterThan task - year-swap corruption ----

class YearSwapCorruption(CorruptionStrategy):
    """Swap start and end years, creating backwards timeline."""

    @property
    def name(self) -> str:
        return "year_swap"

    def corrupt(self, clean_prompt: str, meta: dict) -> str:
        start = meta["start_year"]
        end = meta["end_year"]
        # Replace: swap the two year values
        parts = clean_prompt.split(" to the year ")
        if len(parts) == 2:
            start_part = parts[0].rsplit(" year ", 1)[0]
            return f"{start_part} year {end} to the year {start}."
        return clean_prompt


class GreaterThanGenerator(PromptGenerator):
    """Year-based year-swap corruption task.

    Clean: "The war lasted from the year 1842 to the year 1899."
    y_star: tens digit of start year (e.g., "4" for 1842)
    Corruption: swap start and end years -> "1899 to the year 1842"
    """

    TEMPLATE = "The war lasted from the year {start} to the year {end}."

    YEAR_PAIRS: list[tuple[int, int]] = [
        (1842, 1899), (1776, 1865), (1610, 1789), (1453, 1492),
        (1204, 1291), (1066, 1154), (962, 1024), (800, 843),
        (476, 529), (323, 395), (27, 14), (1, 476),
        (1215, 1250), (1348, 1380), (1453, 1521), (1521, 1648),
        (1618, 1648), (1701, 1789), (1776, 1815), (1803, 1865),
    ]

    def __init__(self, seed: int | None = None):
        super().__init__(seed)
        # Filter: both years must have different tens digits
        self._valid_pairs = [
            (s, e) for s, e in self.YEAR_PAIRS if (s % 100) // 10 != (e % 100) // 10
        ]
        if not self._valid_pairs:
            self._valid_pairs = self.YEAR_PAIRS  # fallback
        self._rng_pair_idx = 0

    @property
    def task_name(self) -> str:
        return "greater_than"

    def generate(self) -> PromptPair:
        # Pick one pair, use it as (start, end) with start < end
        idx = self._rng.randrange(len(self._valid_pairs))
        start, end = self._valid_pairs[idx]
        if start >= end:
            start, end = end, start

        x_cln = self.TEMPLATE.format(start=start, end=end)
        tens = int((start % 100) // 10)
        y_star = str(tens)

        x_crp = YearSwapCorruption().corrupt(x_cln, {"start_year": start, "end_year": end})

        meta = {"start_year": start, "end_year": end, "template": self.TEMPLATE}
        slice_label = SliceLabel(task=self.task_name, corruption="year_swap")

        return PromptPair(
            x_cln=x_cln, x_crp=x_crp, y_star=y_star,
            slice_label=slice_label, meta=meta,
        )


def create_greater_than_dataset(
    n_examples: int, seed: int = 42,
) -> list[PromptPair]:
    """Convenience function to create a GreaterThan dataset."""
    generator = GreaterThanGenerator(seed=seed)
    return generator.generate_batch(n_examples)


# ------ KVRetrieval task - key-swap corruption ----

class KVSwapCorruption(CorruptionStrategy):
    """Replace the target capital with a distractor capital from the context."""

    @property
    def name(self) -> str:
        return "key_swap"

    def corrupt(self, clean_prompt: str, meta: dict) -> str:
        target_country = meta["target_country"]
        distractor_capital = meta["distractor_capital"]
        # Build context manually to avoid accidental double-replacement
        context_lines = meta.get("_context_lines", [])
        query = meta.get("_query", "")
        new_context = []
        for country, capital in context_lines:
            if country == target_country:
                new_context.append(f"{country} is {distractor_capital}.")
            else:
                new_context.append(f"{country} is {capital}.")
        return "\n".join(new_context) + "\n" + query


class KVRetrievalGenerator(PromptGenerator):
    """Country-capital retrieval with in-context lists.

    Clean: "France is Paris. Germany is Berlin. ... Query: What is the capital of Germany?"
    y_star: the correct capital (e.g., "Berlin")
    Corruption: replace the target capital with a distractor from the list
    """

    COUNTRIES: list[tuple[str, str]] = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"),
        ("Spain", "Madrid"), ("Portugal", "Lisbon"), ("Greece", "Athens"),
        ("Poland", "Warsaw"), ("Netherlands", "Amsterdam"), ("Belgium", "Brussels"),
        ("Austria", "Vienna"), ("Czech Republic", "Prague"), ("Hungary", "Budapest"),
        ("Romania", "Bucharest"), ("Bulgaria", "Sofia"), ("Croatia", "Zagreb"),
        ("Serbia", "Belgrade"), ("Denmark", "Copenhagen"), ("Norway", "Oslo"),
        ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Ireland", "Dublin"),
        ("Switzerland", "Bern"), ("Turkey", "Ankara"), ("Egypt", "Cairo"),
        ("Japan", "Tokyo"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
        ("Philippines", "Manila"), ("Indonesia", "Jakarta"),
        ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"),
        ("Myanmar", "Naypyidaw"), ("Kazakhstan", "Astana"),
        ("Ukraine", "Kyiv"), ("Lithuania", "Vilnius"), ("Latvia", "Riga"),
        ("Estonia", "Tallinn"), ("Slovakia", "Bratislava"), ("Slovenia", "Ljubljana"),
        ("Iceland", "Reykjavik"), ("Luxembourg", "Luxembourg"), ("Montenegro", "Podgorica"),
        ("Albania", "Tirana"), ("Cyprus", "Nicosia"),
        ("Moldova", "Chisinau"), ("Georgia", "Tbilisi"), ("Armenia", "Yerevan"),
        ("Azerbaijan", "Baku"), ("Uzbekistan", "Tashkent"), ("Tunisia", "Tunis"),
        ("Morocco", "Rabat"), ("Kenya", "Nairobi"),
        ("Ghana", "Accra"), ("Nigeria", "Abuja"),
        ("Senegal", "Dakar"), ("Tanzania", "Dodoma"),
    ]

    TEMPLATE_QUERY = "Query: What is the capital of {country}?"

    def __init__(
        self,
        n_pairs: int = 10,
        query_index: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(seed)
        self._n_pairs = n_pairs
        self._query_index = query_index

    @property
    def task_name(self) -> str:
        return "kv_retrieval"

    def generate(self) -> PromptPair:
        n = min(self._n_pairs, len(self.COUNTRIES))
        selected = self._rng.sample(self.COUNTRIES, n)

        if self._query_index is not None:
            target_idx = self._query_index % n
        else:
            target_idx = self._rng.randrange(n)

        target_country, target_capital = selected[target_idx]

        other_capitals = [cap for c, cap in self.COUNTRIES if c != target_country]
        distractor_capital = self._rng.choice(other_capitals)

        # Build clean context
        query = self.TEMPLATE_QUERY.format(country=target_country)
        context_str = "\n".join(f"{c} is {cap}." for c, cap in selected)
        x_cln = f"{context_str}\n{query}"

        # Build corrupted context: replace target capital with distractor
        new_context_pairs = [
            (c, cap) if c != target_country else (target_country, distractor_capital)
            for c, cap in selected
        ]
        corrupted_context = "\n".join(f"{c} is {cap}." for c, cap in new_context_pairs)
        x_crp = f"{corrupted_context}\n{query}"

        meta = {
            "target_country": target_country,
            "target_capital": target_capital,
            "distractor_capital": distractor_capital,
            "original_capital": target_capital,
            "_context_lines": selected,
            "_query": query,
        }
        slice_label = SliceLabel(task=self.task_name, corruption="key_swap")

        return PromptPair(
            x_cln=x_cln, x_crp=x_crp, y_star=target_capital,
            slice_label=slice_label, meta=meta,
        )


def create_kv_dataset(
    n_examples: int,
    n_pairs: int = 10,
    query_index: int | None = None,
    seed: int = 42,
) -> list[PromptPair]:
    """Convenience function to create a KVRetrieval dataset."""
    generator = KVRetrievalGenerator(n_pairs=n_pairs, query_index=query_index, seed=seed)
    return generator.generate_batch(n_examples)


# ------ Induction head task ----

INDUCTION_TOKENS: list[str] = [
    "king", "queen", "cat", "dog", "red", "blue", "green", "car", "bus", "sun",
    "moon", "fire", "tree", "rock", "bird", "fish", "star", "rain", "wind", "gold",
    "iron", "wood", "bone", "bell", "door", "flag", "hand", "lamp", "ring", "ship",
    "wall", "wolf", "bear", "deer", "lion", "rose", "leaf", "seed", "sand", "snow",
]

_GAP_TOKENS: list[str] = ["the", "and", "in", "of", "a", "an", "it", "on"]


class InductionTokenSwapCorruption(CorruptionStrategy):
    """Replace second occurrence of A with C to break the induction trigger.

    Clean:      G1 G2 G3 A B G4 G5 G6 A  → predict B
    Corrupted:  G1 G2 G3 A B G4 G5 G6 C  → no induction trigger
    """

    @property
    def name(self) -> str:
        return "token_swap"

    def corrupt(self, clean_prompt: str, meta: dict) -> str:
        token_a = meta["token_a"]
        token_c = meta["token_c"]
        idx = clean_prompt.rfind(" " + token_a)
        if idx == -1:
            return clean_prompt
        return clean_prompt[: idx + 1] + token_c + clean_prompt[idx + 1 + len(token_a) :]


class InductionGenerator(PromptGenerator):
    """Induction head test task — early-trigger variant.

    Generates fixed-9-token prompts of the form:
        G1 G2 G3 A B G4 G5 G6 A   (A at token positions 3 and 8)
    where A and B are single-token content words and G1-G6 are single-token
    function words drawn from disjoint sets. The clean prompt activates GPT-2
    induction heads: after seeing A→B at positions 3-4, the head at position 8
    looks back to position 3, reads B at position 4, and predicts B.

    Corruption ``token_swap``: replace the final A (position 8) with C ≠ A,
    removing the induction trigger. Observable = logit(B) − logit(C), which
    is positive in the clean case and close to zero in the corrupted case.

    See also ``InductionLateGenerator`` for the late-trigger variant (A at
    position 0 instead of 3), which activates the same induction circuit but
    at different (layer, token) positions.
    """

    def __init__(
        self,
        corruption: CorruptionStrategy | None = None,
        seed: int | None = None,
        tokens: list[str] | None = None,
        gap_tokens: list[str] | None = None,
    ):
        super().__init__(seed)
        self._corruption = corruption or InductionTokenSwapCorruption()
        self._tokens = tokens or INDUCTION_TOKENS
        self._gap_tokens = gap_tokens or _GAP_TOKENS

    @property
    def task_name(self) -> str:
        return "induction"

    def generate(self) -> PromptPair:
        token_a, token_b, token_c = self._rng.sample(self._tokens, 3)

        gaps = [self._rng.choice(self._gap_tokens) for _ in range(6)]
        # Template: G1 G2 G3 A B G4 G5 G6 A  (9 tokens, A at positions 3 and 8)
        parts = gaps[:3] + [token_a, token_b] + gaps[3:] + [token_a]
        x_cln = " ".join(parts)

        meta = {"token_a": token_a, "token_b": token_b, "token_c": token_c}
        x_crp = self._corruption.corrupt(x_cln, meta)
        meta["y_distractor"] = token_c  # logit-margin distractor
        slice_label = SliceLabel(task=self.task_name, corruption=self._corruption.name)

        return PromptPair(
            x_cln=x_cln, x_crp=x_crp, y_star=token_b,
            slice_label=slice_label, meta=meta,
        )


class InductionLateGenerator(PromptGenerator):
    """Induction head test task — late-trigger variant.

    Generates fixed-9-token prompts of the form:
        A B G1 G2 G3 G4 G5 G6 A   (A at token positions 0 and 8)
    The induction head at position 8 looks back further (8 positions vs 5 in
    the early variant) to find A at position 0 and read B at position 1.

    Same ``token_swap`` corruption as ``InductionGenerator``: replace the
    final A (position 8) with C. Slice label task = ``induction_late`` so
    classification distinguishes early vs late activation positions.
    """

    def __init__(
        self,
        corruption: CorruptionStrategy | None = None,
        seed: int | None = None,
        tokens: list[str] | None = None,
        gap_tokens: list[str] | None = None,
    ):
        super().__init__(seed)
        self._corruption = corruption or InductionTokenSwapCorruption()
        self._tokens = tokens or INDUCTION_TOKENS
        self._gap_tokens = gap_tokens or _GAP_TOKENS

    @property
    def task_name(self) -> str:
        return "induction_late"

    def generate(self) -> PromptPair:
        token_a, token_b, token_c = self._rng.sample(self._tokens, 3)

        gaps = [self._rng.choice(self._gap_tokens) for _ in range(6)]
        # Template: A B G1 G2 G3 G4 G5 G6 A  (9 tokens, A at positions 0 and 8)
        parts = [token_a, token_b] + gaps + [token_a]
        x_cln = " ".join(parts)

        meta = {"token_a": token_a, "token_b": token_b, "token_c": token_c}
        x_crp = self._corruption.corrupt(x_cln, meta)
        meta["y_distractor"] = token_c
        slice_label = SliceLabel(task=self.task_name, corruption=self._corruption.name)

        return PromptPair(
            x_cln=x_cln, x_crp=x_crp, y_star=token_b,
            slice_label=slice_label, meta=meta,
        )


def create_induction_dataset(
    n_examples: int,
    corruption: str = "token_swap",
    seed: int = 42,
) -> list[PromptPair]:
    """Create an induction dataset.

    ``corruption`` selects both the corruption type and the prompt template:
    - ``token_swap``       : early-trigger variant (A at position 3).
    - ``token_swap_late``  : late-trigger variant (A at position 0).

    Both use the same token-swap corruption (final A → C) and the same
    logit-margin observable logit(B)−logit(C). They produce different slice
    labels (``induction:token_swap`` vs ``induction_late:token_swap``) so that
    binary classification distinguishes the two induction variants.
    """
    _swap = InductionTokenSwapCorruption()
    if corruption == "token_swap":
        generator: PromptGenerator = InductionGenerator(corruption=_swap, seed=seed)
    elif corruption == "token_swap_late":
        generator = InductionLateGenerator(corruption=_swap, seed=seed)
    else:
        raise ValueError(
            f"Unknown corruption: {corruption}. "
            "Available: 'token_swap', 'token_swap_late'."
        )
    return generator.generate_batch(n_examples)


def validate_y_star(tokenizer, pairs: Sequence[PromptPair]) -> dict[str, list[str]]:
    """Validate that y_star is a single token for all pairs.

    Returns:
        Dict mapping task_name -> list of y_star values that are NOT single tokens.
    """
    failures: dict[str, list[str]] = {}
    for pair in pairs:
        tokens = tokenizer.encode(pair.y_star)
        if len(tokens) != 1:
            failures.setdefault(pair.slice_label.task, []).append(pair.y_star)
    return failures
