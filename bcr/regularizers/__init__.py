"""Regularization methods for prediction stability."""

from .bcr import BCRRegularizer, bcr_loss
from .ifr import IFRRegularizer, compute_influence_variance, compute_per_sample_gradients, compute_prediction_jacobian
from .kfac import (
    KFACHooks,
    KFACState,
    compute_grad_covariance_layerwise,
    compute_influence_variance_kfac,
    compute_layerwise_jacobian,
)

__all__ = [
    "BCRRegularizer",
    "bcr_loss",
    "IFRRegularizer",
    "compute_per_sample_gradients",
    "compute_prediction_jacobian",
    "compute_influence_variance",
    "KFACState",
    "KFACHooks",
    "compute_layerwise_jacobian",
    "compute_influence_variance_kfac",
    "compute_grad_covariance_layerwise",
]
