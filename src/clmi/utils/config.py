"""Typed configuration objects for CLMI experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


FTMode = Literal["full", "lora"]
MitigationMode = Literal["none", "freeze_nc", "anchor_reg"]
InterventionMode = Literal["reinforce", "suppress"]


@dataclass(frozen=True)
class DataConfig:
    """Synthetic task generation configuration."""

    n_keys: int = 400
    n_values: int = 400
    overlap: float = 0.0
    prompt_template: str = "Q: {X}\nA:"
    max_answer_tokens: int = 2


@dataclass(frozen=True)
class TrainConfig:
    """Fine-tuning configuration for one continual-learning run."""

    model_name: str = "gpt2"
    device: str = "auto"
    seed: int = 42
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    batch_size: int = 8
    max_steps: int = 200
    warmup_steps: int = 0
    eval_every: int = 50
    save_every: int = 100
    epochs_a: int = 1
    epochs_b: int = 1
    epochs_a2: int = 1
    loss_mask_answer_only: bool = True
    ft_mode: FTMode = "full"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("c_attn", "c_proj", "c_fc")
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class SubspaceConfig:
    """Layer-wise subspace extraction parameters."""

    k: int = 16
    ridge_alpha: float = 1.0
    bootstrap_iters: int = 32
    layers: tuple[int, ...] = ()
    alpha_layers: tuple[float, ...] = ()


@dataclass(frozen=True)
class InterventionConfig:
    """Projection intervention parameters used for functional NC."""

    beta: float = 0.2
    mode: InterventionMode = "reinforce"
    layers: tuple[int, ...] = ()


@dataclass(frozen=True)
class RobustnessConfig:
    """Noise robustness evaluation parameters."""

    typo_probability: float = 0.1
    token_flip_probability: float = 0.05
    embedding_noise_std: float = 0.01
    n_samples: int = 64


@dataclass(frozen=True)
class MitigationConfig:
    """Mitigation strategy configuration during task-B training."""

    mode: MitigationMode = "none"
    top_nc_layers: int = 2
    anchor_weight: float = 0.1
    anchor_samples: int = 64


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config for a pair run.

    ``overlap`` and ``seed`` are convenience properties that delegate to
    ``data.overlap`` and ``train.seed`` respectively.
    """

    run_id: str
    pair_id: int
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    subspace: SubspaceConfig = field(default_factory=SubspaceConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)

    @property
    def overlap(self) -> float:
        return self.data.overlap

    @property
    def seed(self) -> int:
        return self.train.seed


DEFAULT_OVERLAPS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
