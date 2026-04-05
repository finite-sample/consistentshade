"""Influence Function Regularization.

This module implements IFR with fixes for:
1. Jacobian computation using torch.autograd.grad (not .backward())
2. Correct influence variance formula with proper matrix operations
3. Reasonable Fisher initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_per_sample_gradients(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, loss_fn):
    """
    Compute per-sample gradients without polluting model gradients.

    Uses torch.autograd.grad to avoid side effects on model.grad attributes.
    Returns detached gradients for use in IF penalty computation.
    """
    grads = []
    params = [p for p in model.parameters() if p.requires_grad]

    for i in range(len(xb)):
        pred = model(xb[i : i + 1])
        loss = loss_fn(pred, yb[i : i + 1])

        grad = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False)
        grad_flat = torch.cat([g.detach().flatten() for g in grad])
        grads.append(grad_flat)

    return torch.stack(grads)


def compute_prediction_jacobian(model: nn.Module, x: torch.Tensor, out_dim: int = 1):
    """
    Compute Jacobian of predictions w.r.t. parameters.

    Uses torch.autograd.grad instead of .backward() to avoid polluting gradients.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    pred = model(x.unsqueeze(0))

    if out_dim == 1:
        grad = torch.autograd.grad(pred.sum(), params, create_graph=False, retain_graph=False)
        jac = torch.cat([g.flatten() for g in grad])
        return jac.unsqueeze(0)
    else:
        jacs = []
        for k in range(out_dim):
            grad = torch.autograd.grad(
                pred[0, k],
                params,
                create_graph=False,
                retain_graph=(k < out_dim - 1),
            )
            jac = torch.cat([g.flatten() for g in grad])
            jacs.append(jac)
        return torch.stack(jacs)


def compute_influence_variance(jac: torch.Tensor, h_inv: torch.Tensor, grad_cov_diag: torch.Tensor):
    """
    Compute influence-based prediction variance estimate.

    For diagonal Hessian approximation:
        sigma^2(x) ~ J^T H^{-1} Sigma_g H^{-1} J

    With diagonal approximation:
        sigma^2(x) ~ sum_d (J_d^2 * H_d^{-2} * Sigma_g_d)
    """
    if jac.dim() == 1:
        return (jac**2 * h_inv**2 * grad_cov_diag).sum()
    else:
        return ((jac**2 * h_inv**2 * grad_cov_diag).sum(dim=1)).mean()


class IFRRegularizer:
    """
    Influence Function Regularization for prediction stability.

    Approximates: sigma^2_boot(x) ~ J_x^T H^{-1} Sigma_g H^{-1} J_x
    Using diagonal Fisher approximation for H.
    """

    def __init__(
        self,
        model: nn.Module,
        lam: float = 0.1,
        fisher_ema: float = 0.99,
        n_jacobian_samples: int = 8,
        fisher_init: float = 1e-2,
        warmup_epochs: int = 2,
    ):
        """
        Initialize IFR regularizer.

        Args:
            model: Neural network model
            lam: Regularization strength
            fisher_ema: EMA coefficient for Fisher diagonal
            n_jacobian_samples: Number of samples for Jacobian computation
            fisher_init: Initial value for Fisher diagonal
            warmup_epochs: Number of epochs before applying IFR penalty
        """
        self.model = model
        self.lam = lam
        self.fisher_ema = fisher_ema
        self.n_jacobian_samples = n_jacobian_samples
        self.warmup_epochs = warmup_epochs

        n_params = sum(p.numel() for p in model.parameters())
        self.fisher_diag = torch.ones(n_params) * fisher_init
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Set current epoch for warmup tracking."""
        self.current_epoch = epoch

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn,
        out_dim: int = 1,
    ):
        """
        Compute IFR loss with influence-based variance penalty.

        Args:
            x: Input features (B, d_in)
            y: Targets
            loss_fn: Loss function (should work with per-sample computation)
            out_dim: Output dimension (1 for regression, n_classes for classification)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        B = len(x)

        preds = self.model(x)
        if out_dim == 1:
            sup_loss = F.mse_loss(preds, y)
        else:
            sup_loss = F.cross_entropy(preds, y)

        if self.current_epoch >= self.warmup_epochs and self.lam > 0:
            per_sample_grads = compute_per_sample_gradients(self.model, x, y, loss_fn)

            grad_cov_diag = per_sample_grads.var(dim=0)
            self.fisher_diag = (
                self.fisher_ema * self.fisher_diag.to(x.device)
                + (1 - self.fisher_ema) * (per_sample_grads**2).mean(dim=0)
            )

            sample_idx = torch.randperm(B)[: min(self.n_jacobian_samples, B)]
            if_vars = []
            for idx in sample_idx:
                jac = compute_prediction_jacobian(self.model, x[idx], out_dim=out_dim)
                jac = jac.squeeze().detach()
                h_inv = 1.0 / (self.fisher_diag + 1e-6)
                if_var = compute_influence_variance(jac, h_inv, grad_cov_diag)
                if_vars.append(if_var)

            if_penalty = torch.stack(if_vars).mean()
            total_loss = sup_loss + self.lam * if_penalty
            if_penalty_val = if_penalty.item()
        else:
            total_loss = sup_loss
            if_penalty_val = 0.0

        metrics = {
            "supervised_loss": sup_loss.item(),
            "if_penalty": if_penalty_val,
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction."""
        self.model.eval()
        return self.model(x)
