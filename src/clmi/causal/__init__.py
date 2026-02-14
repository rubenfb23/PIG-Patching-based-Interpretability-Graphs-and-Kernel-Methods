"""Causal subspace extraction and intervention routines."""

from .logit_diff import compute_logit_diff
from .patching import (
    compute_functional_kl_ab_ba,
    compute_intervention_effect_embedding,
)
from .subspaces import (
    ProjectorPack,
    build_task_projectors,
    collect_layer_activations,
    sample_random_projectors,
)

__all__ = [
    "compute_logit_diff",
    "collect_layer_activations",
    "build_task_projectors",
    "ProjectorPack",
    "compute_functional_kl_ab_ba",
    "compute_intervention_effect_embedding",
    "sample_random_projectors",
]
