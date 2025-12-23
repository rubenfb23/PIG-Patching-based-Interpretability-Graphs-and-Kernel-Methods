"""Model setup and activation hooking for patching experiments.

This module provides utilities for:
- Loading a HuggingFace transformer model with frozen weights
- Capturing residual, attention-head, and MLP activations
- Replacing activations during forward passes (patching)
- Computing observables (target-token logits)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

NODE_TYPE_RES = "res"
NODE_TYPE_MLP = "mlp"
NODE_TYPE_ATT = "att"
ALLOWED_NODE_TYPES = {NODE_TYPE_RES, NODE_TYPE_MLP, NODE_TYPE_ATT}


@dataclass
class ActivationCache:
    """Cache for storing activations from a forward pass.

    Stores activations indexed by (layer, token, node_type, head) positions.
    """

    activations: dict[tuple[int, int, str, Optional[int]], Tensor] = field(
        default_factory=dict
    )

    def store(
        self,
        layer: int,
        token: int,
        activation: Tensor,
        node_type: str = NODE_TYPE_RES,
        head: Optional[int] = None,
    ) -> None:
        """Store activation for a given (layer, token, node_type, head)."""
        self.activations[(layer, token, node_type, head)] = (
            activation.detach().clone()
        )

    def get(
        self,
        layer: int,
        token: int,
        node_type: str = NODE_TYPE_RES,
        head: Optional[int] = None,
    ) -> Optional[Tensor]:
        """Retrieve activation for a given (layer, token, node_type, head)."""
        return self.activations.get((layer, token, node_type, head))

    def get_all_for_layer(
        self,
        layer: int,
        node_type: str = NODE_TYPE_RES,
        head: Optional[int] = None,
    ) -> dict[int, Tensor]:
        """Get all activations for a specific layer and node type."""
        return {
            token: act
            for (l, token, ntype, nhead), act in self.activations.items()
            if l == layer and ntype == node_type and nhead == head
        }

    def clear(self) -> None:
        """Clear all cached activations."""
        self.activations.clear()

    def __len__(self) -> int:
        return len(self.activations)


class HookedModel:
    """A wrapper around a HuggingFace model with activation hooking capabilities.

    Supports:
    - Capturing residual, attention-head, and MLP activations
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
        self._patch_positions: set[tuple[int, int, str, Optional[int]]] = set()
        self._capture_components: set[str] = {NODE_TYPE_RES}
        self._patch_components: set[str] = set()

    def _setup_architecture_info(self) -> None:
        """Extract architecture information from the model."""
        config = self.model.config
        self.n_layers = config.n_layer
        self.n_heads = config.n_head
        self.d_model = config.n_embd
        self.head_dim = self.d_model // self.n_heads
        self.vocab_size = config.vocab_size

    def _get_residual_hook_points(self) -> list[str]:
        """Get the names of residual stream hook points for GPT-2."""
        # For GPT-2, residual stream is after each transformer block
        # The blocks are at model.transformer.h[i]
        return [f"transformer.h.{i}" for i in range(self.n_layers)]

    def _make_capture_hook(
        self, layer: int, node_type: str
    ) -> Callable[[torch.nn.Module, tuple, tuple], None]:
        """Create a hook that captures activations."""
        def hook(module: torch.nn.Module, inputs: tuple, output: tuple) -> None:
            if self._capture_cache is None:
                return
            # GPT-2 block output is a tuple: (hidden_states, present, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # hidden_states shape: [batch, seq_len, d_model]
            seq_len = hidden_states.shape[1]
            for token in range(seq_len):
                self._capture_cache.store(
                    layer,
                    token,
                    hidden_states[0, token, :],
                    node_type=node_type,
                )
        return hook

    def _make_patch_hook(
        self, layer: int, node_type: str
    ) -> Callable[[torch.nn.Module, tuple, tuple], tuple | Tensor]:
        """Create a hook that patches (replaces) activations."""
        def hook(
            module: torch.nn.Module, inputs: tuple, output: tuple
        ) -> tuple | Tensor:
            if self._patch_cache is None or not self._patch_positions:
                return output
            if isinstance(output, tuple):
                hidden_states = output[0].clone()
                tail = output[1:]
            else:
                hidden_states = output.clone()
                tail = None
            for token in range(hidden_states.shape[1]):
                if (layer, token, node_type, None) in self._patch_positions:
                    patch_act = self._patch_cache.get(
                        layer, token, node_type=node_type
                    )
                    if patch_act is not None:
                        hidden_states[0, token, :] = patch_act.to(
                            hidden_states.device
                        )
            if tail is None:
                return hidden_states
            return (hidden_states,) + tail
        return hook

    def _make_attn_pre_hook(
        self,
        layer: int,
        capture: bool,
        patch: bool,
    ) -> Callable[[torch.nn.Module, tuple], Optional[tuple]]:
        """Create a pre-hook for attention head inputs to c_proj."""
        def hook(module: torch.nn.Module, inputs: tuple) -> Optional[tuple]:
            if not inputs:
                return None
            hidden_states = inputs[0]
            batch, seq_len, _ = hidden_states.shape

            if capture and self._capture_cache is not None:
                heads = hidden_states.reshape(
                    batch, seq_len, self.n_heads, self.head_dim
                )
                for token in range(seq_len):
                    for head in range(self.n_heads):
                        self._capture_cache.store(
                            layer,
                            token,
                            heads[0, token, head, :],
                            node_type=NODE_TYPE_ATT,
                            head=head,
                        )

            if patch and self._patch_cache is not None and self._patch_positions:
                updated = hidden_states.clone()
                heads = updated.view(
                    batch, seq_len, self.n_heads, self.head_dim
                )
                for token in range(seq_len):
                    for head in range(self.n_heads):
                        if (layer, token, NODE_TYPE_ATT, head) in self._patch_positions:
                            patch_act = self._patch_cache.get(
                                layer,
                                token,
                                node_type=NODE_TYPE_ATT,
                                head=head,
                            )
                            if patch_act is not None:
                                heads[0, token, head, :] = patch_act.to(
                                    updated.device
                                )
                return (updated,) + inputs[1:]

            return None
        return hook

    def _normalize_patch_node(
        self, patch_node: tuple
    ) -> tuple[int, int, str, Optional[int]]:
        """Normalize patch node tuples to (layer, token, node_type, head)."""
        if len(patch_node) == 2:
            layer, token = patch_node
            return (layer, token, NODE_TYPE_RES, None)
        if len(patch_node) == 3:
            layer, token, node_type = patch_node
            if node_type == NODE_TYPE_ATT:
                raise ValueError("Attention patch nodes require head index")
            return (layer, token, node_type, None)
        if len(patch_node) == 4:
            layer, token, node_type, head = patch_node
            if node_type == NODE_TYPE_ATT and head is None:
                raise ValueError("Attention patch nodes require head index")
            if node_type == NODE_TYPE_ATT and not (0 <= head < self.n_heads):
                raise ValueError(f"Invalid attention head index: {head}")
            return (layer, token, node_type, head)
        raise ValueError(f"Invalid patch node format: {patch_node}")

    def _normalize_patch_nodes(
        self, patch_nodes: Iterable[tuple]
    ) -> set[tuple[int, int, str, Optional[int]]]:
        return {self._normalize_patch_node(node) for node in patch_nodes}

    def _validate_node_types(self, node_types: Iterable[str]) -> None:
        invalid = set(node_types) - ALLOWED_NODE_TYPES
        if invalid:
            raise ValueError(f"Unsupported node types: {sorted(invalid)}")

    def _register_hooks(
        self,
        capture: bool = False,
        patch: bool = False,
    ) -> None:
        """Register forward hooks for selected components."""
        self._clear_hooks()

        for layer in range(self.n_layers):
            # Access the transformer block
            block = self.model.transformer.h[layer]

            if capture and NODE_TYPE_RES in self._capture_components:
                hook = block.register_forward_hook(
                    self._make_capture_hook(layer, NODE_TYPE_RES)
                )
                self._hooks.append(hook)

            if patch and NODE_TYPE_RES in self._patch_components:
                hook = block.register_forward_hook(
                    self._make_patch_hook(layer, NODE_TYPE_RES)
                )
                self._hooks.append(hook)

            if capture and NODE_TYPE_MLP in self._capture_components:
                hook = block.mlp.register_forward_hook(
                    self._make_capture_hook(layer, NODE_TYPE_MLP)
                )
                self._hooks.append(hook)

            if patch and NODE_TYPE_MLP in self._patch_components:
                hook = block.mlp.register_forward_hook(
                    self._make_patch_hook(layer, NODE_TYPE_MLP)
                )
                self._hooks.append(hook)

            if capture and NODE_TYPE_ATT in self._capture_components:
                hook = block.attn.c_proj.register_forward_pre_hook(
                    self._make_attn_pre_hook(
                        layer, capture=True, patch=False
                    )
                )
                self._hooks.append(hook)

            if patch and NODE_TYPE_ATT in self._patch_components:
                hook = block.attn.c_proj.register_forward_pre_hook(
                    self._make_attn_pre_hook(
                        layer, capture=False, patch=True
                    )
                )
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
        return self.cache_clean_components(prompt, node_types=None)

    @torch.no_grad()
    def cache_clean_components(
        self, prompt: str, node_types: Optional[Iterable[str]] = None
    ) -> ActivationCache:
        """Run a forward pass and cache activations for selected components.

        Args:
            prompt: The clean prompt to cache activations for
            node_types: Iterable of component types ("res", "mlp", "att").
                If None, defaults to residual stream only.

        Returns:
            ActivationCache containing activations at all requested positions
        """
        cache = ActivationCache()
        self._capture_cache = cache
        self._capture_components = (
            set(node_types) if node_types is not None else {NODE_TYPE_RES}
        )
        if not self._capture_components:
            raise ValueError("node_types must include at least one component")
        self._validate_node_types(self._capture_components)

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
        patch_node: tuple,
    ) -> float:
        """Compute observable with patched activations.

        Args:
            prompt: The (corrupted) prompt to run
            target_token: The target token to get logit for
            cache: Cache containing clean activations to patch in
            patch_node: (layer, token[, node_type[, head]]) position to patch

        Returns:
            The logit value after patching
        """
        self._patch_cache = cache
        normalized = self._normalize_patch_node(patch_node)
        self._patch_positions = {normalized}
        self._patch_components = {normalized[2]}
        self._validate_node_types(self._patch_components)

        self._register_hooks(capture=False, patch=True)

        input_ids = self.tokenize(prompt)
        logits = self.forward(input_ids)

        self._clear_hooks()
        self._patch_cache = None
        self._patch_positions.clear()
        self._patch_components.clear()

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
        patch_nodes: set[tuple],
    ) -> float:
        """Compute observable with multiple positions patched.

        Args:
            prompt: The (corrupted) prompt to run
            target_token: The target token to get logit for
            cache: Cache containing clean activations to patch in
            patch_nodes: Set of (layer, token[, node_type[, head]]) positions

        Returns:
            The logit value after patching
        """
        self._patch_cache = cache
        self._patch_positions = self._normalize_patch_nodes(patch_nodes)
        self._patch_components = {node[2] for node in self._patch_positions}
        self._validate_node_types(self._patch_components)

        self._register_hooks(capture=False, patch=True)

        input_ids = self.tokenize(prompt)
        logits = self.forward(input_ids)

        self._clear_hooks()
        self._patch_cache = None
        self._patch_positions.clear()
        self._patch_components.clear()

        target_id = self.get_token_id(target_token)
        last_pos_logits = logits[0, -1, :]

        return last_pos_logits[target_id].item()

    def get_num_layers(self) -> int:
        """Return the number of transformer layers."""
        return self.n_layers

    def get_num_tokens(self, prompt: str) -> int:
        """Return the number of tokens in a prompt."""
        return len(self.tokenize(prompt)[0])
