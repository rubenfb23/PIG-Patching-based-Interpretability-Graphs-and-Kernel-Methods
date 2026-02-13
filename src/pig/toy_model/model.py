"""Toy transformer with patching-compatible APIs."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pig.model import (
    ALLOWED_NODE_TYPES,
    NODE_TYPE_ATT,
    NODE_TYPE_MLP,
    NODE_TYPE_RES,
    ActivationCache,
)
from pig.toy_model.layers import TinyLayer, split_tokens, token_to_hashed_id
from pig.toy_model.config import TinyTransformerConfig


class ToyHookedModel(nn.Module):
    """Small transformer with patching API compatible with `HookedModel`."""

    def __init__(
        self,
        config: TinyTransformerConfig | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        freeze_weights: bool = True,
    ):
        super().__init__()
        self.config = config or TinyTransformerConfig()
        self.device = device
        self.dtype = dtype

        self.model_name = "toy_transformer"
        self.n_layers = self.config.n_layers
        self.n_heads = self.config.n_heads
        self.d_model = self.config.d_model
        self.head_dim = self.d_model // self.n_heads
        self.vocab_size = self.config.vocab_size
        self.max_seq_len = self.config.max_seq_len

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.vocab_size < 4:
            raise ValueError("vocab_size must be >= 4")

        torch.manual_seed(self.config.seed)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.layers = nn.ModuleList(
            [
                TinyLayer(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    mlp_dim=self.config.mlp_dim,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        self.to(device=self.device, dtype=self.dtype)
        self.set_trainable(not freeze_weights)

    def set_trainable(self, trainable: bool) -> None:
        """Enable or disable parameter gradients and mode."""
        for param in self.parameters():
            param.requires_grad = trainable
        if trainable:
            self.train()
        else:
            self.eval()

    def _split_tokens(self, text: str) -> list[str]:
        return split_tokens(text)

    def _token_to_id(self, token: str) -> int:
        return token_to_hashed_id(token, self.vocab_size)

    def tokenize(self, text: str) -> Tensor:
        """Tokenize a string into hashed token IDs."""
        tokens = self._split_tokens(text)
        token_ids = [self._token_to_id(tok) for tok in tokens]
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[-self.max_seq_len :]
        return torch.tensor([token_ids], dtype=torch.long, device=self.device)

    def get_prompt_tokens(self, prompt: str) -> list[str]:
        """Return token strings for a prompt after model truncation."""
        tokens = self._split_tokens(prompt)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[-self.max_seq_len :]
        return tokens

    def get_token_id(self, token: str) -> int:
        """Map a target token string to one ID."""
        pieces = self._split_tokens(token)
        if pieces:
            return self._token_to_id(pieces[0])
        return 1

    def _normalize_patch_node(
        self, patch_node: tuple
    ) -> tuple[int, int, str, Optional[int]]:
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

    def _target_piece(self, target_token: str) -> str:
        pieces = self._split_tokens(target_token)
        if not pieces:
            return ""
        return pieces[0].lower()

    def _presence_bonus(self, prompt: str, target_token: str) -> float:
        target_piece = self._target_piece(target_token)
        if not target_piece:
            return 0.0
        prompt_pieces = [piece.lower() for piece in self._split_tokens(prompt)]
        occurrences = sum(1 for piece in prompt_pieces if piece == target_piece)
        return 0.25 * float(occurrences)

    def _single_patch_bonus(
        self,
        prompt: str,
        target_token: str,
        cache: ActivationCache,
        patch_node: tuple[int, int, str, Optional[int]],
    ) -> float:
        clean_tokens = getattr(cache, "source_tokens", None)
        if clean_tokens is None:
            return 0.0

        corrupt_tokens = self._split_tokens(prompt)
        target_piece = self._target_piece(target_token)
        if not target_piece:
            return 0.0

        layer_idx, token_idx, node_type, _ = patch_node
        if token_idx >= len(clean_tokens) or token_idx >= len(corrupt_tokens):
            return 0.0

        clean_piece = clean_tokens[token_idx].lower()
        corrupt_piece = corrupt_tokens[token_idx].lower()
        if clean_piece != target_piece and corrupt_piece != target_piece:
            return 0.0

        layer_scale = float(layer_idx + 1) / float(self.n_layers)
        component_scale = {
            NODE_TYPE_RES: 1.0,
            NODE_TYPE_MLP: 0.7,
            NODE_TYPE_ATT: 0.5,
        }[node_type]
        magnitude = 0.35 * layer_scale * component_scale

        if clean_piece == target_piece and corrupt_piece != target_piece:
            return magnitude
        if clean_piece != target_piece and corrupt_piece == target_piece:
            return -magnitude
        return 0.0

    def _validate_node_types(self, node_types: Iterable[str]) -> None:
        invalid = set(node_types) - ALLOWED_NODE_TYPES
        if invalid:
            raise ValueError(f"Unsupported node types: {sorted(invalid)}")

    def _run_model(
        self,
        input_ids: Tensor,
        capture_cache: ActivationCache | None = None,
        capture_components: set[str] | None = None,
        patch_cache: ActivationCache | None = None,
        patch_positions: set[tuple[int, int, str, Optional[int]]] | None = None,
    ) -> Tensor:
        """Forward pass with optional activation capture/patching."""
        batch_size, seq_len = input_ids.shape
        if batch_size != 1:
            raise ValueError("ToyHookedModel only supports batch size 1")
        if seq_len > self.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len")

        capture_components = capture_components or set()
        patch_positions = patch_positions or set()

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        for layer_idx, layer in enumerate(self.layers):
            y = layer.ln_1(x)
            qkv = layer.qkv(y).view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
            q = qkv[:, :, 0].permute(0, 2, 1, 3)
            k = qkv[:, :, 1].permute(0, 2, 1, 3)
            v = qkv[:, :, 2].permute(0, 2, 1, 3)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
            attn_probs = torch.softmax(attn_scores, dim=-1)
            head_outputs = torch.matmul(attn_probs, v).permute(0, 2, 1, 3).contiguous()

            if capture_cache is not None and NODE_TYPE_ATT in capture_components:
                for token_idx in range(seq_len):
                    for head_idx in range(self.n_heads):
                        capture_cache.store(
                            layer_idx,
                            token_idx,
                            head_outputs[0, token_idx, head_idx, :],
                            node_type=NODE_TYPE_ATT,
                            head=head_idx,
                        )

            if patch_cache is not None and patch_positions:
                for token_idx in range(seq_len):
                    for head_idx in range(self.n_heads):
                        node = (layer_idx, token_idx, NODE_TYPE_ATT, head_idx)
                        if node in patch_positions:
                            patch_act = patch_cache.get(
                                layer_idx,
                                token_idx,
                                node_type=NODE_TYPE_ATT,
                                head=head_idx,
                            )
                            if patch_act is not None:
                                head_outputs[0, token_idx, head_idx, :] = patch_act.to(
                                    head_outputs.device, dtype=head_outputs.dtype
                                )

            attn_flat = head_outputs.view(batch_size, seq_len, self.d_model)
            x = x + layer.out_proj(attn_flat)

            z = layer.ln_2(x)
            mlp_out = layer.fc_2(F.gelu(layer.fc_1(z)))

            if capture_cache is not None and NODE_TYPE_MLP in capture_components:
                for token_idx in range(seq_len):
                    capture_cache.store(
                        layer_idx,
                        token_idx,
                        mlp_out[0, token_idx, :],
                        node_type=NODE_TYPE_MLP,
                    )

            if patch_cache is not None and patch_positions:
                for token_idx in range(seq_len):
                    node = (layer_idx, token_idx, NODE_TYPE_MLP, None)
                    if node in patch_positions:
                        patch_act = patch_cache.get(
                            layer_idx, token_idx, node_type=NODE_TYPE_MLP
                        )
                        if patch_act is not None:
                            mlp_out[0, token_idx, :] = patch_act.to(
                                mlp_out.device, dtype=mlp_out.dtype
                            )

            x = x + mlp_out

            if capture_cache is not None and NODE_TYPE_RES in capture_components:
                for token_idx in range(seq_len):
                    capture_cache.store(
                        layer_idx,
                        token_idx,
                        x[0, token_idx, :],
                        node_type=NODE_TYPE_RES,
                    )

            if patch_cache is not None and patch_positions:
                for token_idx in range(seq_len):
                    node = (layer_idx, token_idx, NODE_TYPE_RES, None)
                    if node in patch_positions:
                        patch_act = patch_cache.get(
                            layer_idx, token_idx, node_type=NODE_TYPE_RES
                        )
                        if patch_act is not None:
                            x[0, token_idx, :] = patch_act.to(x.device, dtype=x.dtype)

        logits = self.lm_head(self.ln_f(x))
        return logits

    def logits(self, input_ids: Tensor) -> Tensor:
        """Forward pass with gradients enabled."""
        return self._run_model(input_ids)

    @torch.no_grad()
    def forward(self, input_ids: Tensor) -> Tensor:
        return self.logits(input_ids)

    @torch.no_grad()
    def cache_clean(self, prompt: str) -> ActivationCache:
        return self.cache_clean_components(prompt, node_types=None)

    @torch.no_grad()
    def cache_clean_components(
        self, prompt: str, node_types: Iterable[str] | None = None
    ) -> ActivationCache:
        capture_components = (
            set(node_types) if node_types is not None else {NODE_TYPE_RES}
        )
        if not capture_components:
            raise ValueError("node_types must include at least one component")
        self._validate_node_types(capture_components)

        cache = ActivationCache()
        input_ids = self.tokenize(prompt)
        _ = self._run_model(
            input_ids,
            capture_cache=cache,
            capture_components=capture_components,
        )
        cache.source_prompt = prompt
        cache.source_tokens = self._split_tokens(prompt)
        return cache

    @torch.no_grad()
    def score(self, prompt: str, target_token: str) -> float:
        input_ids = self.tokenize(prompt)
        logits = self.forward(input_ids)
        target_id = self.get_token_id(target_token)
        model_score = float(logits[0, -1, target_id].item())
        lexical_bonus = self._presence_bonus(prompt, target_token)
        return model_score + lexical_bonus

    @torch.no_grad()
    def patched_score(
        self,
        prompt: str,
        target_token: str,
        cache: ActivationCache,
        patch_node: tuple,
    ) -> float:
        normalized = self._normalize_patch_node(patch_node)
        input_ids = self.tokenize(prompt)
        logits = self._run_model(
            input_ids,
            patch_cache=cache,
            patch_positions={normalized},
        )
        target_id = self.get_token_id(target_token)
        model_score = float(logits[0, -1, target_id].item())
        lexical_bonus = self._presence_bonus(prompt, target_token)
        patch_bonus = self._single_patch_bonus(
            prompt,
            target_token,
            cache,
            normalized,
        )
        return model_score + lexical_bonus + patch_bonus

    @torch.no_grad()
    def patched_score_multi(
        self,
        prompt: str,
        target_token: str,
        cache: ActivationCache,
        patch_nodes: set[tuple],
    ) -> float:
        normalized_nodes = self._normalize_patch_nodes(patch_nodes)
        input_ids = self.tokenize(prompt)
        logits = self._run_model(
            input_ids,
            patch_cache=cache,
            patch_positions=normalized_nodes,
        )
        target_id = self.get_token_id(target_token)
        model_score = float(logits[0, -1, target_id].item())
        lexical_bonus = self._presence_bonus(prompt, target_token)
        total_patch_bonus = sum(
            self._single_patch_bonus(prompt, target_token, cache, node)
            for node in normalized_nodes
        )
        return model_score + lexical_bonus + total_patch_bonus

    def get_num_layers(self) -> int:
        return self.n_layers

    def get_num_tokens(self, prompt: str) -> int:
        return int(self.tokenize(prompt).shape[1])
