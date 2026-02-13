"""Configuration objects for the toy model package."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TinyTransformerConfig:
    """Configuration for the tiny transformer model."""

    vocab_size: int = 512
    max_seq_len: int = 128
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    mlp_dim: int = 128
    seed: int = 0


@dataclass(frozen=True)
class TinyTrainingConfig:
    """Training hyperparameters for `ToyHookedModel`."""

    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    shuffle: bool = True
    seed: int = 0
