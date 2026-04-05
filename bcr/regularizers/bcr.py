"""Bootstrap Consistency Regularization."""

import itertools
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCRRegularizer:
    """
    Bootstrap Consistency Regularization for prediction stability.

    This class wraps K models and provides methods for computing BCR loss
    and making ensemble predictions. Use this for custom training loops.

    Example:
        bcr = BCRRegularizer(your_model, K=3, lam=0.05)
        for xb, yb in dataloader:
            loss, metrics = bcr.compute_loss(xb, yb, loss_fn=nn.MSELoss(reduction='none'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        preds = bcr.predict(test_x)
    """

    def __init__(
        self,
        model_or_factory: Union[nn.Module, Callable[[], nn.Module]],
        K: int = 3,
        lam: float = 0.05,
    ):
        """
        Initialize BCR regularizer.

        Args:
            model_or_factory: Either a model instance (will create K copies) or
                            a factory function that returns a new model instance.
            K: Number of bootstrap models to use.
            lam: Regularization strength for variance penalty.
        """
        self.K = K
        self.lam = lam

        if callable(model_or_factory) and not isinstance(model_or_factory, nn.Module):
            self.models = nn.ModuleList([model_or_factory() for _ in range(K)])
        else:
            self.models = nn.ModuleList([model_or_factory] + [self._clone_model(model_or_factory) for _ in range(K - 1)])

    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a fresh copy of the model with reinitialized weights."""
        import copy

        clone = copy.deepcopy(model)
        for m in clone.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        return clone

    def parameters(self):
        """Return all parameters from all models."""
        return itertools.chain(*(m.parameters() for m in self.models))

    def train(self):
        """Set all models to training mode."""
        for m in self.models:
            m.train()

    def eval(self):
        """Set all models to evaluation mode."""
        for m in self.models:
            m.eval()

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute BCR loss with bootstrap weighting and variance penalty.

        Args:
            x: Input features (B, d_in)
            y: Targets (B,) for regression, (B,) for classification
            loss_fn: Loss function that accepts (predictions, targets) and returns
                    per-sample losses (reduction='none' or similar).
            reduction: How to reduce the final loss ('mean' or 'sum')

        Returns:
            Tuple of (total_loss, metrics_dict) where metrics_dict contains
            'supervised_loss' and 'variance_penalty'.
        """
        B = len(x)
        preds = []
        sup_losses = []

        for m in self.models:
            px = m(x)
            preds.append(px)

            weights = torch.poisson(torch.ones(B, device=x.device))
            per_sample_loss = loss_fn(px, y)
            weighted_loss = (weights * per_sample_loss).mean()
            sup_losses.append(weighted_loss)

        preds = torch.stack(preds)
        sup_loss = torch.stack(sup_losses).mean()

        if preds.dim() == 3:
            probs = F.softmax(preds, dim=-1)
            var_pen = probs.var(dim=0).mean()
        else:
            var_pen = preds.var(dim=0, unbiased=False).mean()

        total_loss = sup_loss + self.lam * var_pen

        metrics = {
            "supervised_loss": sup_loss.item(),
            "variance_penalty": var_pen.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make ensemble prediction by averaging across all K models.

        Args:
            x: Input features (B, d_in)

        Returns:
            Ensemble predictions (B,) or (B, n_classes) for classification
        """
        self.eval()
        preds = torch.stack([m(x) for m in self.models])
        return preds.mean(0)


def bcr_loss(
    models: List[nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lam: float = 0.05,
) -> Tuple[torch.Tensor, dict]:
    """
    Functional API for computing BCR loss.

    Args:
        models: List of K models
        x: Input features (B, d_in)
        y: Targets
        loss_fn: Per-sample loss function
        lam: Variance penalty weight

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    B = len(x)
    preds = []
    sup_losses = []

    for m in models:
        px = m(x)
        preds.append(px)

        weights = torch.poisson(torch.ones(B, device=x.device))
        per_sample_loss = loss_fn(px, y)
        weighted_loss = (weights * per_sample_loss).mean()
        sup_losses.append(weighted_loss)

    preds = torch.stack(preds)
    sup_loss = torch.stack(sup_losses).mean()

    if preds.dim() == 3:
        probs = F.softmax(preds, dim=-1)
        var_pen = probs.var(dim=0).mean()
    else:
        var_pen = preds.var(dim=0, unbiased=False).mean()

    total_loss = sup_loss + lam * var_pen

    metrics = {
        "supervised_loss": sup_loss.item(),
        "variance_penalty": var_pen.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, metrics
