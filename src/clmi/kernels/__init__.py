"""Task-level kernels and downstream utilities."""

from .curriculum import greedy_curriculum
from .kernels import build_kernel_matrix, k_func, k_nc, k_proj
from .predict import fit_kernel_ridge_predictor

__all__ = [
    "k_proj",
    "k_nc",
    "k_func",
    "build_kernel_matrix",
    "greedy_curriculum",
    "fit_kernel_ridge_predictor",
]
