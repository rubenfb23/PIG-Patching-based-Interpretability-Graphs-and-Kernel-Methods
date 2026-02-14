"""Synthetic continual-learning task generation."""

from .synth_tasks import TaskMapping, TaskPair, XYExample, generate_task_pair
from .token_utils import select_clean_tokens

__all__ = [
    "XYExample",
    "TaskMapping",
    "TaskPair",
    "generate_task_pair",
    "select_clean_tokens",
]
