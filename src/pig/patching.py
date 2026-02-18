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
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from pig.model import (
    NODE_TYPE_ATT,
    NODE_TYPE_MLP,
    NODE_TYPE_RES,
    ALLOWED_NODE_TYPES,
    HookedModel,
)
from pig.prompts import PromptPair, SliceLabel


@dataclass(frozen=True)
class ComponentSpec:
    """Component spec for a per-token patch node axis."""

    node_type: str
    head: Optional[int] = None

    def to_dict(self) -> dict:
        return {"type": self.node_type, "head": self.head}

    @classmethod
    def from_dict(cls, data: dict) -> "ComponentSpec":
        return cls(node_type=data["type"], head=data.get("head"))


def build_component_axis(
    node_types: Sequence[str], num_heads: int
) -> list[ComponentSpec]:
    """Build a component axis list for patching."""
    node_type_set = set(node_types)
    invalid = node_type_set - ALLOWED_NODE_TYPES
    if invalid:
        raise ValueError(f"Unsupported node types: {sorted(invalid)}")
    if not node_type_set:
        raise ValueError("node_types must include at least one component")

    axis: list[ComponentSpec] = []
    if NODE_TYPE_ATT in node_type_set:
        for head in range(num_heads):
            axis.append(ComponentSpec(node_type=NODE_TYPE_ATT, head=head))
    if NODE_TYPE_MLP in node_type_set:
        axis.append(ComponentSpec(node_type=NODE_TYPE_MLP))
    if NODE_TYPE_RES in node_type_set:
        axis.append(ComponentSpec(node_type=NODE_TYPE_RES))
    return axis


@dataclass
class PatchEffectTensor:
    """Dense tensor of patch effects for a single example.

    Stores E_u for all nodes u = (layer, token, component) in a dense tensor.

    Attributes:
        effects: Dense tensor of shape [num_layers, num_tokens, num_components]
        component_axis: Component spec list shared across all tensors
        prompt_pair: The original prompt pair used
        base_score: The baseline score O(x^crp) before patching
        clean_score: The clean score O(x^cln)
        token_labels: Human-readable token strings for the clean prompt
    """

    effects: NDArray[np.float32]
    component_axis: list[ComponentSpec]
    prompt_pair: PromptPair
    base_score: float
    clean_score: float
    token_labels: list[str] = field(default_factory=list)

    @property
    def num_layers(self) -> int:
        return self.effects.shape[0]

    @property
    def num_tokens(self) -> int:
        return self.effects.shape[1]

    @property
    def num_components(self) -> int:
        return self.effects.shape[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.effects.shape

    def component_index(self, node_type: str, head: Optional[int] = None) -> int:
        """Get the component axis index for a node type and head."""
        for idx, spec in enumerate(self.component_axis):
            if spec.node_type == node_type and spec.head == head:
                return idx
        raise ValueError(f"Component not found: type={node_type}, head={head}")

    def get_effect(self, layer: int, token: int, component_idx: int = 0) -> float:
        """Get the patch effect at a specific (layer, token, component)."""
        return float(self.effects[layer, token, component_idx])

    def get_significant_positions(
        self, threshold: float = 0.1
    ) -> list[tuple[int, int, int, float]]:
        """Get positions with |effect| above threshold.

        Returns:
            List of (layer, token, component_idx, effect) tuples
            sorted by |effect|.
        """
        positions = []
        for layer in range(self.num_layers):
            for token in range(self.num_tokens):
                for comp_idx in range(self.num_components):
                    effect = self.effects[layer, token, comp_idx]
                    if abs(effect) > threshold:
                        positions.append((layer, token, comp_idx, float(effect)))
        return sorted(positions, key=lambda x: abs(x[3]), reverse=True)

    def to_dict(self) -> dict:
        """Serialize to dictionary for caching."""
        return {
            "effects": self.effects.tolist(),
            "component_axis": [c.to_dict() for c in self.component_axis],
            "prompt_pair": self.prompt_pair.to_dict(),
            "base_score": self.base_score,
            "clean_score": self.clean_score,
            "token_labels": self.token_labels,
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
            component_axis=[
                ComponentSpec.from_dict(c)
                for c in data.get(
                    "component_axis", [{"type": NODE_TYPE_RES, "head": None}]
                )
            ],
            prompt_pair=prompt_pair,
            base_score=data["base_score"],
            clean_score=data["clean_score"],
            token_labels=data.get("token_labels", []),
        )


@dataclass
class PatchEffectDataset:
    """Collection of patch-effect tensors for multiple examples.

    Organizes tensors by slice for efficient access during graph construction.
    """

    tensors: list[PatchEffectTensor] = field(default_factory=list)

    def add(self, tensor: PatchEffectTensor) -> None:
        """Add a tensor to the dataset."""
        if self.tensors:
            expected = self.tensors[0].component_axis
            if tensor.component_axis != expected:
                raise ValueError("Inconsistent component_axis in dataset")
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
            Matrix of shape [num_examples, num_layers * max_tokens * num_components]
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
            truncated = t.effects[:, :max_tokens, :]
            flattened.append(truncated.reshape(-1))

        return np.stack(flattened, axis=0)

    def get_common_dimensions(self, slice_label: SliceLabel) -> tuple[int, int, int]:
        """Get the common (num_layers, min_tokens, num_components) for a slice.

        Returns dimensions that work for all examples in the slice.
        """
        tensors = self.get_by_slice(slice_label)
        if not tensors:
            raise ValueError(f"No tensors found for slice {slice_label}")

        num_layers = tensors[0].num_layers
        min_tokens = min(t.num_tokens for t in tensors)
        num_components = tensors[0].num_components

        return num_layers, min_tokens, num_components

    def get_component_axis(self) -> list[ComponentSpec]:
        """Return the shared component axis for the dataset."""
        if not self.tensors:
            return []
        return self.tensors[0].component_axis

    def compute_statistics(self) -> dict:
        """Compute summary statistics for the dataset."""
        if not self.tensors:
            return {}

        all_effects = np.concatenate([t.effects.reshape(-1) for t in self.tensors])

        stats = {
            "num_examples": len(self.tensors),
            "num_slices": len(self.get_slices()),
            "effect_mean": float(np.mean(all_effects)),
            "effect_std": float(np.std(all_effects)),
            "effect_min": float(np.min(all_effects)),
            "effect_max": float(np.max(all_effects)),
            "significant_positions_mean": np.mean(
                [len(t.get_significant_positions(0.1)) for t in self.tensors]
            ),
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

    def _compute_key(
        self,
        prompt_pair: PromptPair,
        model_name: str,
        component_axis: Optional[Sequence[ComponentSpec]],
    ) -> str:
        """Compute a unique cache key for a prompt pair."""
        content = json.dumps(
            {
                "x_cln": prompt_pair.x_cln,
                "x_crp": prompt_pair.x_crp,
                "y_star": prompt_pair.y_star,
                "slice": str(prompt_pair.slice_label),
                "model": model_name,
                "components": (
                    [c.to_dict() for c in component_axis]
                    if component_axis is not None
                    else None
                ),
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(
        self,
        prompt_pair: PromptPair,
        model_name: str,
        component_axis: Optional[Sequence[ComponentSpec]] = None,
    ) -> Optional[PatchEffectTensor]:
        """Retrieve a cached tensor if it exists."""
        key = self._compute_key(prompt_pair, model_name, component_axis)
        cache_path = self.cache_dir / f"{key}.json"

        if cache_path.exists():
            with open(cache_path, "r") as f:
                data = json.load(f)
            return PatchEffectTensor.from_dict(data)
        return None

    def put(self, tensor: PatchEffectTensor, model_name: str) -> None:
        """Store a tensor in the cache."""
        key = self._compute_key(tensor.prompt_pair, model_name, tensor.component_axis)
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
        node_types: Optional[Sequence[str]] = None,
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
        self.node_types = (
            tuple(node_types) if node_types is not None else (NODE_TYPE_RES,)
        )
        self.component_axis = build_component_axis(self.node_types, self.model.n_heads)

    def compute_single(self, prompt_pair: PromptPair) -> PatchEffectTensor:
        """Compute patch-effect tensor for a single example.

        Args:
            prompt_pair: The clean/corrupted prompt pair

        Returns:
            PatchEffectTensor with effects for all (layer, token, component) positions
        """
        # Check cache first
        if self.cache is not None:
            cached = self.cache.get(
                prompt_pair,
                self.model.model_name,
                self.component_axis,
            )
            if cached is not None:
                return cached

        # Cache clean activations
        clean_cache = self.model.cache_clean_components(
            prompt_pair.x_cln, node_types=self.node_types
        )

        # Compute baseline scores
        clean_score = self.model.score(prompt_pair.x_cln, prompt_pair.y_star)
        base_score = self.model.score(prompt_pair.x_crp, prompt_pair.y_star)

        # Get dimensions
        num_layers = self.model.get_num_layers()
        num_tokens = self.model.get_num_tokens(prompt_pair.x_crp)

        # Compute effects for all positions
        num_components = len(self.component_axis)
        effects = np.zeros((num_layers, num_tokens, num_components), dtype=np.float32)

        for layer in range(num_layers):
            for token in range(num_tokens):
                for comp_idx, comp in enumerate(self.component_axis):
                    if comp.node_type == NODE_TYPE_ATT:
                        patch_node = (layer, token, NODE_TYPE_ATT, comp.head)
                    else:
                        patch_node = (layer, token, comp.node_type)
                    patched_score = self.model.patched_score(
                        prompt_pair.x_crp,
                        prompt_pair.y_star,
                        clean_cache,
                        patch_node,
                    )
                    effects[layer, token, comp_idx] = patched_score - base_score

        token_labels = self.model.get_prompt_tokens(prompt_pair.x_cln)

        tensor = PatchEffectTensor(
            effects=effects,
            component_axis=self.component_axis,
            prompt_pair=prompt_pair,
            base_score=base_score,
            clean_score=clean_score,
            token_labels=token_labels,
        )

        # Store in cache
        if self.cache is not None:
            self.cache.put(tensor, self.model.model_name)

        return tensor

    def compute_batch(self, prompt_pairs: list[PromptPair]) -> PatchEffectDataset:
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
    node_types: Optional[Sequence[str]] = None,
) -> PatchEffectDataset:
    """Convenience function to compute patch effects for a list of prompts.

    Args:
        model: The hooked model to use
        prompt_pairs: List of prompt pairs to process
        cache_dir: Optional cache directory (None to disable caching)
        show_progress: Whether to print progress
        node_types: Iterable of component types ("res", "mlp", "att")

    Returns:
        PatchEffectDataset with all computed tensors
    """
    cache = PatchEffectCache(cache_dir) if cache_dir else None

    def progress_callback(current: int, total: int) -> None:
        if show_progress:
            print(f"\r  Computing patch effects: {current}/{total}", end="")

    computer = PatchEffectComputer(
        model,
        cache,
        progress_callback,
        node_types=node_types,
    )
    dataset = computer.compute_batch(prompt_pairs)

    if show_progress:
        print()  # Newline after progress

    return dataset
