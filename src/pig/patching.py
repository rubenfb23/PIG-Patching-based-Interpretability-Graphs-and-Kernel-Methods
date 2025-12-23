"""Patch-effect tensor computation and caching.

This module provides functionality for computing and storing patch-effect tensors
E_u^(i) for all examples and nodes, enabling efficient graph construction.

The patch effect E_u = O(patched_forward(x^crp; u)) - O(forward(x^crp))
measures the causal effect of patching node u on the model's output.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from pig.model import HookedModel
from pig.prompts import PromptPair, SliceLabel


@dataclass
class PatchEffectTensor:
    """Dense tensor of patch effects for a single example.

    Stores E_u for all nodes u = (layer, token) in a dense matrix.

    Attributes:
        effects: Dense matrix of shape [num_layers, num_tokens]
        prompt_pair: The original prompt pair used
        base_score: The baseline score O(x^crp) before patching
        clean_score: The clean score O(x^cln)
    """

    effects: NDArray[np.float32]
    prompt_pair: PromptPair
    base_score: float
    clean_score: float

    @property
    def num_layers(self) -> int:
        return self.effects.shape[0]

    @property
    def num_tokens(self) -> int:
        return self.effects.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.effects.shape

    def get_effect(self, layer: int, token: int) -> float:
        """Get the patch effect at a specific (layer, token) position."""
        return float(self.effects[layer, token])

    def get_significant_positions(
        self, threshold: float = 0.1
    ) -> list[tuple[int, int, float]]:
        """Get positions with |effect| above threshold.

        Returns:
            List of (layer, token, effect) tuples sorted by |effect|.
        """
        positions = []
        for layer in range(self.num_layers):
            for token in range(self.num_tokens):
                effect = self.effects[layer, token]
                if abs(effect) > threshold:
                    positions.append((layer, token, float(effect)))
        return sorted(positions, key=lambda x: abs(x[2]), reverse=True)

    def to_dict(self) -> dict:
        """Serialize to dictionary for caching."""
        return {
            "effects": self.effects.tolist(),
            "prompt_pair": self.prompt_pair.to_dict(),
            "base_score": self.base_score,
            "clean_score": self.clean_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PatchEffectTensor":
        """Deserialize from dictionary."""
        from pig.prompts import PromptPair, SliceLabel

        prompt_dict = data["prompt_pair"]
        slice_label = SliceLabel(
            task=prompt_dict["slice"]["task"],
            corruption=prompt_dict["slice"]["corruption"],
        )
        prompt_pair = PromptPair(
            x_cln=prompt_dict["x_cln"],
            x_crp=prompt_dict["x_crp"],
            y_star=prompt_dict["y_star"],
            slice_label=slice_label,
            meta=prompt_dict["meta"],
        )
        return cls(
            effects=np.array(data["effects"], dtype=np.float32),
            prompt_pair=prompt_pair,
            base_score=data["base_score"],
            clean_score=data["clean_score"],
        )


@dataclass
class PatchEffectDataset:
    """Collection of patch-effect tensors for multiple examples.

    Organizes tensors by slice for efficient access during graph construction.
    """

    tensors: list[PatchEffectTensor] = field(default_factory=list)

    def add(self, tensor: PatchEffectTensor) -> None:
        """Add a tensor to the dataset."""
        self.tensors.append(tensor)

    def __len__(self) -> int:
        return len(self.tensors)

    def __iter__(self):
        return iter(self.tensors)

    def __getitem__(self, idx: int) -> PatchEffectTensor:
        return self.tensors[idx]

    def get_by_slice(self, slice_label: SliceLabel) -> list[PatchEffectTensor]:
        """Get all tensors for a specific slice."""
        return [t for t in self.tensors if t.prompt_pair.slice_label == slice_label]

    def get_slices(self) -> list[SliceLabel]:
        """Get all unique slice labels in the dataset."""
        return list(set(t.prompt_pair.slice_label for t in self.tensors))

    def get_effect_matrix(
        self, slice_label: SliceLabel, max_tokens: Optional[int] = None
    ) -> NDArray[np.float32]:
        """Get stacked effect matrix for a slice.

        Args:
            slice_label: The slice to get effects for
            max_tokens: If specified, truncate to this many tokens.
                       If None, uses the minimum token count in the slice.

        Returns:
            Matrix of shape [num_examples, num_layers * max_tokens]
        """
        tensors = self.get_by_slice(slice_label)
        if not tensors:
            raise ValueError(f"No tensors found for slice {slice_label}")

        # Determine the number of tokens to use (minimum across all examples)
        if max_tokens is None:
            max_tokens = min(t.num_tokens for t in tensors)

        # Truncate each tensor to max_tokens and flatten
        flattened = []
        for t in tensors:
            truncated = t.effects[:, :max_tokens]
            flattened.append(truncated.flatten())

        return np.stack(flattened, axis=0)

    def get_common_dimensions(
        self, slice_label: SliceLabel
    ) -> tuple[int, int]:
        """Get the common (num_layers, min_tokens) for a slice.

        Returns dimensions that work for all examples in the slice.
        """
        tensors = self.get_by_slice(slice_label)
        if not tensors:
            raise ValueError(f"No tensors found for slice {slice_label}")

        num_layers = tensors[0].num_layers
        min_tokens = min(t.num_tokens for t in tensors)

        return num_layers, min_tokens

    def compute_statistics(self) -> dict:
        """Compute summary statistics for the dataset."""
        if not self.tensors:
            return {}

        all_effects = np.concatenate([t.effects.flatten() for t in self.tensors])

        stats = {
            "num_examples": len(self.tensors),
            "num_slices": len(self.get_slices()),
            "effect_mean": float(np.mean(all_effects)),
            "effect_std": float(np.std(all_effects)),
            "effect_min": float(np.min(all_effects)),
            "effect_max": float(np.max(all_effects)),
            "significant_positions_mean": np.mean([
                len(t.get_significant_positions(0.1)) for t in self.tensors
            ]),
        }

        return stats


class PatchEffectCache:
    """Cache for storing and retrieving patch-effect tensors.

    Uses file-based caching with content-addressable hashing.
    """

    def __init__(self, cache_dir: Path | str = ".cache/patch_effects"):
        """Initialize the cache.

        Args:
            cache_dir: Directory for storing cached tensors.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, prompt_pair: PromptPair, model_name: str) -> str:
        """Compute a unique cache key for a prompt pair."""
        content = json.dumps({
            "x_cln": prompt_pair.x_cln,
            "x_crp": prompt_pair.x_crp,
            "y_star": prompt_pair.y_star,
            "slice": str(prompt_pair.slice_label),
            "model": model_name,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(
        self, prompt_pair: PromptPair, model_name: str
    ) -> Optional[PatchEffectTensor]:
        """Retrieve a cached tensor if it exists."""
        key = self._compute_key(prompt_pair, model_name)
        cache_path = self.cache_dir / f"{key}.json"

        if cache_path.exists():
            with open(cache_path, "r") as f:
                data = json.load(f)
            return PatchEffectTensor.from_dict(data)
        return None

    def put(
        self, tensor: PatchEffectTensor, model_name: str
    ) -> None:
        """Store a tensor in the cache."""
        key = self._compute_key(tensor.prompt_pair, model_name)
        cache_path = self.cache_dir / f"{key}.json"

        with open(cache_path, "w") as f:
            json.dump(tensor.to_dict(), f)

    def clear(self) -> int:
        """Clear all cached tensors. Returns count of deleted files."""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count


class PatchEffectComputer:
    """Computes patch-effect tensors efficiently.

    Handles the computation of E_u^(i) for all positions in an example,
    with optional caching to avoid redundant computation.
    """

    def __init__(
        self,
        model: HookedModel,
        cache: Optional[PatchEffectCache] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Initialize the computer.

        Args:
            model: The hooked model to use for patching
            cache: Optional cache for storing computed tensors
            progress_callback: Optional callback for progress updates (current, total)
        """
        self.model = model
        self.cache = cache
        self.progress_callback = progress_callback

    def compute_single(self, prompt_pair: PromptPair) -> PatchEffectTensor:
        """Compute patch-effect tensor for a single example.

        Args:
            prompt_pair: The clean/corrupted prompt pair

        Returns:
            PatchEffectTensor with effects for all (layer, token) positions
        """
        # Check cache first
        if self.cache is not None:
            cached = self.cache.get(prompt_pair, self.model.model_name)
            if cached is not None:
                return cached

        # Cache clean activations
        clean_cache = self.model.cache_clean(prompt_pair.x_cln)

        # Compute baseline scores
        clean_score = self.model.score(prompt_pair.x_cln, prompt_pair.y_star)
        base_score = self.model.score(prompt_pair.x_crp, prompt_pair.y_star)

        # Get dimensions
        num_layers = self.model.get_num_layers()
        num_tokens = self.model.get_num_tokens(prompt_pair.x_crp)

        # Compute effects for all positions
        effects = np.zeros((num_layers, num_tokens), dtype=np.float32)

        for layer in range(num_layers):
            for token in range(num_tokens):
                patched_score = self.model.patched_score(
                    prompt_pair.x_crp,
                    prompt_pair.y_star,
                    clean_cache,
                    (layer, token),
                )
                effects[layer, token] = patched_score - base_score

        tensor = PatchEffectTensor(
            effects=effects,
            prompt_pair=prompt_pair,
            base_score=base_score,
            clean_score=clean_score,
        )

        # Store in cache
        if self.cache is not None:
            self.cache.put(tensor, self.model.model_name)

        return tensor

    def compute_batch(
        self, prompt_pairs: list[PromptPair]
    ) -> PatchEffectDataset:
        """Compute patch-effect tensors for a batch of examples.

        Args:
            prompt_pairs: List of prompt pairs to process

        Returns:
            PatchEffectDataset containing all computed tensors
        """
        dataset = PatchEffectDataset()

        for i, pair in enumerate(prompt_pairs):
            if self.progress_callback is not None:
                self.progress_callback(i + 1, len(prompt_pairs))

            tensor = self.compute_single(pair)
            dataset.add(tensor)

        return dataset


def compute_patch_effects(
    model: HookedModel,
    prompt_pairs: list[PromptPair],
    cache_dir: Optional[str] = None,
    show_progress: bool = True,
) -> PatchEffectDataset:
    """Convenience function to compute patch effects for a list of prompts.

    Args:
        model: The hooked model to use
        prompt_pairs: List of prompt pairs to process
        cache_dir: Optional cache directory (None to disable caching)
        show_progress: Whether to print progress

    Returns:
        PatchEffectDataset with all computed tensors
    """
    cache = PatchEffectCache(cache_dir) if cache_dir else None

    def progress_callback(current: int, total: int) -> None:
        if show_progress:
            print(f"\r  Computing patch effects: {current}/{total}", end="")

    computer = PatchEffectComputer(model, cache, progress_callback)
    dataset = computer.compute_batch(prompt_pairs)

    if show_progress:
        print()  # Newline after progress

    return dataset
