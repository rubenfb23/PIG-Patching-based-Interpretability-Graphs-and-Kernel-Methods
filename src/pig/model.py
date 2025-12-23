"""Model setup and activation hooking for patching experiments.

This module provides utilities for:
- Loading a HuggingFace transformer model with frozen weights
- Capturing residual stream activations at any (layer, token) position
- Replacing activations during forward passes (patching)
- Computing observables (target-token logits)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


@dataclass
class ActivationCache:
    """Cache for storing activations from a forward pass.

    Stores residual stream activations indexed by (layer, token) positions.
    """

    activations: dict[tuple[int, int], Tensor] = field(default_factory=dict)

    def store(self, layer: int, token: int, activation: Tensor) -> None:
        """Store activation for a given (layer, token) position."""
        self.activations[(layer, token)] = activation.detach().clone()

    def get(self, layer: int, token: int) -> Optional[Tensor]:
        """Retrieve activation for a given (layer, token) position."""
        return self.activations.get((layer, token))

    def get_all_for_layer(self, layer: int) -> dict[int, Tensor]:
        """Get all activations for a specific layer."""
        return {
            token: act for (l, token), act in self.activations.items() if l == layer
        }

    def clear(self) -> None:
        """Clear all cached activations."""
        self.activations.clear()

    def __len__(self) -> int:
        return len(self.activations)


class HookedModel:
    """A wrapper around a HuggingFace model with activation hooking capabilities.

    Supports:
    - Capturing residual stream activations at any layer
    - Patching (replacing) activations during forward passes
    - Computing target-token logits as observables
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the hooked model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on (None for auto-detect, "cpu" to force CPU)
            dtype: Data type for model weights
        """
        self.dtype = dtype
        self.model_name = model_name

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Determine device - try GPU first, fall back to CPU
        if device is None:
            if torch.cuda.is_available():
                try:
                    # Try loading on GPU
                    self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                        model_name,
                    ).to("cuda")
                    device = "cuda"
                except (RuntimeError, torch.OutOfMemoryError):
                    # Fall back to CPU if GPU OOM
                    self.model = AutoModelForCausalLM.from_pretrained(model_name)
                    device = "cpu"
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                device = "cpu"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        self.device = device

        # Freeze model weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Get model architecture info
        self._setup_architecture_info()

        # Hook management
        self._hooks: list = []
        self._capture_cache: Optional[ActivationCache] = None
        self._patch_cache: Optional[ActivationCache] = None
        self._patch_positions: set[tuple[int, int]] = set()

    def _setup_architecture_info(self) -> None:
        """Extract architecture information from the model."""
        config = self.model.config
        self.n_layers = config.n_layer
        self.n_heads = config.n_head
        self.d_model = config.n_embd
        self.vocab_size = config.vocab_size

    def _get_residual_hook_points(self) -> list[str]:
        """Get the names of residual stream hook points for GPT-2."""
        # For GPT-2, residual stream is after each transformer block
        # The blocks are at model.transformer.h[i]
        return [f"transformer.h.{i}" for i in range(self.n_layers)]

    def _make_capture_hook(
        self, layer: int
    ) -> Callable[[torch.nn.Module, tuple, tuple], None]:
        """Create a hook that captures activations."""
        def hook(module: torch.nn.Module, inputs: tuple, output: tuple) -> None:
            if self._capture_cache is not None:
                # GPT-2 block output is a tuple: (hidden_states, present, ...)
                # hidden_states shape: [batch, seq_len, d_model]
                hidden_states = output[0]
                seq_len = hidden_states.shape[1]
                for token in range(seq_len):
                    self._capture_cache.store(
                        layer, token, hidden_states[0, token, :]
                    )
        return hook

    def _make_patch_hook(
        self, layer: int
    ) -> Callable[[torch.nn.Module, tuple, tuple], tuple]:
        """Create a hook that patches (replaces) activations."""
        def hook(
            module: torch.nn.Module, inputs: tuple, output: tuple
        ) -> tuple:
            if self._patch_cache is not None and self._patch_positions:
                # GPT-2 block output is a tuple: (hidden_states, present, ...)
                hidden_states = output[0].clone()
                for token in range(hidden_states.shape[1]):
                    if (layer, token) in self._patch_positions:
                        patch_act = self._patch_cache.get(layer, token)
                        if patch_act is not None:
                            hidden_states[0, token, :] = patch_act.to(hidden_states.device)
                # Return modified tuple
                return (hidden_states,) + output[1:]
            return output
        return hook

    def _register_hooks(
        self,
        capture: bool = False,
        patch: bool = False,
    ) -> None:
        """Register forward hooks on residual stream points."""
        self._clear_hooks()

        for layer in range(self.n_layers):
            # Access the transformer block
            block = self.model.transformer.h[layer]

            if capture:
                hook = block.register_forward_hook(self._make_capture_hook(layer))
                self._hooks.append(hook)

            if patch:
                hook = block.register_forward_hook(self._make_patch_hook(layer))
                self._hooks.append(hook)

    def _clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def tokenize(self, text: str) -> Tensor:
        """Tokenize text and return input IDs tensor."""
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)

    def get_token_id(self, token: str) -> int:
        """Get the token ID for a given token string."""
        # Handle tokens that may need space prefix
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
        # Try with space prefix (common for GPT-2)
        ids_with_space = self.tokenizer.encode(" " + token, add_special_tokens=False)
        if len(ids_with_space) == 1:
            return ids_with_space[0]
        return ids[0]  # Return first token if multi-token

    @torch.no_grad()
    def forward(self, input_ids: Tensor) -> Tensor:
        """Run a forward pass and return logits."""
        outputs = self.model(input_ids)
        return outputs.logits

    @torch.no_grad()
    def cache_clean(self, prompt: str) -> ActivationCache:
        """Run a forward pass and cache all residual stream activations.

        Args:
            prompt: The clean prompt to cache activations for

        Returns:
            ActivationCache containing activations at all (layer, token) positions
        """
        cache = ActivationCache()
        self._capture_cache = cache

        self._register_hooks(capture=True, patch=False)

        input_ids = self.tokenize(prompt)
        _ = self.forward(input_ids)

        self._clear_hooks()
        self._capture_cache = None

        return cache

    @torch.no_grad()
    def score(self, prompt: str, target_token: str) -> float:
        """Compute the observable O(x) = logit for target token.

        Args:
            prompt: The input prompt
            target_token: The target token to get logit for

        Returns:
            The logit value for the target token at the last position
        """
        input_ids = self.tokenize(prompt)
        logits = self.forward(input_ids)

        # Get logit for target token at last position
        target_id = self.get_token_id(target_token)
        last_pos_logits = logits[0, -1, :]  # [vocab_size]

        return last_pos_logits[target_id].item()

    @torch.no_grad()
    def patched_score(
        self,
        prompt: str,
        target_token: str,
        cache: ActivationCache,
        patch_node: tuple[int, int],
    ) -> float:
        """Compute observable with patched activations.

        Args:
            prompt: The (corrupted) prompt to run
            target_token: The target token to get logit for
            cache: Cache containing clean activations to patch in
            patch_node: (layer, token) position to patch

        Returns:
            The logit value after patching
        """
        self._patch_cache = cache
        self._patch_positions = {patch_node}

        self._register_hooks(capture=False, patch=True)

        input_ids = self.tokenize(prompt)
        logits = self.forward(input_ids)

        self._clear_hooks()
        self._patch_cache = None
        self._patch_positions.clear()

        # Get logit for target token at last position
        target_id = self.get_token_id(target_token)
        last_pos_logits = logits[0, -1, :]

        return last_pos_logits[target_id].item()

    @torch.no_grad()
    def patched_score_multi(
        self,
        prompt: str,
        target_token: str,
        cache: ActivationCache,
        patch_nodes: set[tuple[int, int]],
    ) -> float:
        """Compute observable with multiple positions patched.

        Args:
            prompt: The (corrupted) prompt to run
            target_token: The target token to get logit for
            cache: Cache containing clean activations to patch in
            patch_nodes: Set of (layer, token) positions to patch

        Returns:
            The logit value after patching
        """
        self._patch_cache = cache
        self._patch_positions = patch_nodes

        self._register_hooks(capture=False, patch=True)

        input_ids = self.tokenize(prompt)
        logits = self.forward(input_ids)

        self._clear_hooks()
        self._patch_cache = None
        self._patch_positions.clear()

        target_id = self.get_token_id(target_token)
        last_pos_logits = logits[0, -1, :]

        return last_pos_logits[target_id].item()

    def get_num_layers(self) -> int:
        """Return the number of transformer layers."""
        return self.n_layers

    def get_num_tokens(self, prompt: str) -> int:
        """Return the number of tokens in a prompt."""
        return len(self.tokenize(prompt)[0])
