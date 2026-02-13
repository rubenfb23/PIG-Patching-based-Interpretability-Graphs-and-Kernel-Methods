"""Public API for the local toy transformer package."""

from typing import TYPE_CHECKING

from pig.toy_model.config import TinyTrainingConfig, TinyTransformerConfig
from pig.toy_model.model import ToyHookedModel

if TYPE_CHECKING:
    from pig.toy_model.trainer import EpochMetrics

__all__ = [
    "EpochMetrics",
    "TinyTrainingConfig",
    "TinyTransformerConfig",
    "ToyHookedModel",
    "train_toy_model",
]


def __getattr__(name: str):
    if name in {"EpochMetrics", "train_toy_model"}:
        from pig.toy_model.trainer import EpochMetrics, train_toy_model

        exports = {
            "EpochMetrics": EpochMetrics,
            "train_toy_model": train_toy_model,
        }
        return exports[name]
    raise AttributeError(f"module 'pig.toy_model' has no attribute {name!r}")
