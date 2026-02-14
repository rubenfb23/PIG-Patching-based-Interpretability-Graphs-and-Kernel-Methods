"""Random seed utilities for deterministic execution."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch.

    Args:
        seed: Base seed.
        deterministic: If True, request deterministic torch algorithms.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
