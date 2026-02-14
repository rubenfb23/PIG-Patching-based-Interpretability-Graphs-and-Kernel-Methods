"""Metrics for forgetting, non-commutativity, robustness and statistics."""

from .forgetting import compute_forgetting, compute_hysteresis_metrics, compute_weight_distance
from .noncommutativity import (
    compute_jensen_shannon,
    compute_operator_noncommutativity,
    compute_prob_kl,
)
from .robustness import evaluate_robustness
from .statistics import (
    bootstrap_mean_ci,
    compute_correlation_table,
    paired_permutation_test,
    pearson_with_ci,
    spearman_with_ci,
)

__all__ = [
    "bootstrap_mean_ci",
    "compute_correlation_table",
    "compute_forgetting",
    "compute_hysteresis_metrics",
    "compute_jensen_shannon",
    "compute_operator_noncommutativity",
    "compute_prob_kl",
    "compute_weight_distance",
    "evaluate_robustness",
    "paired_permutation_test",
    "pearson_with_ci",
    "spearman_with_ci",
]
