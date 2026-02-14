"""Model loading, training and hook-based interventions."""

from .finetune import (
    FinetuneResult,
    evaluate_task_accuracy,
    finetune_on_task,
    freeze_transformer_layers,
    save_checkpoint,
    unfreeze_transformer_layers,
)
from .gpt2_loader import load_model_tokenizer, resolve_device
from .hooks import run_with_projection_intervention

__all__ = [
    "load_model_tokenizer",
    "resolve_device",
    "finetune_on_task",
    "evaluate_task_accuracy",
    "save_checkpoint",
    "FinetuneResult",
    "run_with_projection_intervention",
    "freeze_transformer_layers",
    "unfreeze_transformer_layers",
]
