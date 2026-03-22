"""Influence Function Regularization training.

This module contains the corrected IFR implementation with fixes for:
1. Jacobian computation using torch.autograd.grad (not .backward())
2. Correct influence variance formula with proper matrix operations
3. Reasonable Fisher initialization
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from ..config import CLASSIFICATION_CONFIG, IFR_CONFIG, REGRESSION_CONFIG
from ..models import DropMLP
from ..utils import set_seed


def compute_per_sample_gradients(model, xb, yb, loss_fn):
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


def compute_prediction_jacobian(model, x, out_dim=1):
    """
    Compute Jacobian of predictions w.r.t. parameters.

    FIX: Uses torch.autograd.grad instead of .backward() to avoid
    polluting gradients for main optimization.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    pred = model(x.unsqueeze(0))

    if out_dim == 1:
        grad = torch.autograd.grad(
            pred.sum(), params, create_graph=False, retain_graph=False
        )
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


def compute_influence_variance(jac, h_inv, grad_cov_diag):
    """
    Compute influence-based prediction variance estimate.

    For diagonal Hessian approximation:
        σ²(x) ≈ J^T H^{-1} Σ_g H^{-1} J

    With diagonal approximation:
        σ²(x) ≈ Σ_d (J_d² * H_d^{-2} * Σ_g_d)

    FIX: The original code did element-wise multiplication incorrectly.
    This version correctly computes the quadratic form.
    """
    if jac.dim() == 1:
        return (jac**2 * h_inv**2 * grad_cov_diag).sum()
    else:
        return ((jac**2 * h_inv**2 * grad_cov_diag).sum(dim=1)).mean()


def train_ifr_regression(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    lam=None,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
    fisher_ema=None,
    n_jacobian_samples=None,
):
    """
    Influence Function Regularization for regression.

    Approximates: σ²_boot(x) ≈ J_x^T H^{-1} Σ_g H^{-1} J_x
    Using diagonal Fisher approximation for H.

    FIXES:
    1. Uses torch.autograd.grad to avoid gradient pollution
    2. Correct influence variance formula
    3. Reasonable Fisher initialization (1e-2 instead of 1e-4)
    """
    cfg_reg = REGRESSION_CONFIG
    cfg_ifr = IFR_CONFIG
    lam = lam if lam is not None else cfg_ifr["lam"]
    fisher_ema = fisher_ema if fisher_ema is not None else cfg_ifr["fisher_ema"]
    n_jacobian_samples = (
        n_jacobian_samples
        if n_jacobian_samples is not None
        else cfg_ifr["n_jacobian_samples"]
    )
    epochs = epochs or cfg_reg["epochs"]
    bs = bs or cfg_reg["batch_size"]
    lr = lr or cfg_reg["lr"]
    hid = hid or cfg_reg["hidden_dim"]
    dropout = dropout or cfg_reg["dropout"]

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    n_params = sum(p.numel() for p in model.parameters())
    fisher_diag = torch.ones(n_params) * cfg_ifr["fisher_init"]

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            B = len(xb)

            preds = model(xb)
            sup_loss = F.mse_loss(preds, yb)

            if epoch >= 2 and lam > 0:
                per_sample_grads = compute_per_sample_gradients(
                    model, xb, yb, lambda p, y: F.mse_loss(p.view(-1), y.view(-1))
                )

                grad_cov_diag = per_sample_grads.var(dim=0)
                fisher_diag = (
                    fisher_ema * fisher_diag
                    + (1 - fisher_ema) * (per_sample_grads**2).mean(dim=0)
                )

                sample_idx = torch.randperm(B)[: min(n_jacobian_samples, B)]
                if_vars = []
                for idx in sample_idx:
                    jac = compute_prediction_jacobian(model, xb[idx], out_dim=1)
                    jac = jac.squeeze().detach()
                    h_inv = 1.0 / (fisher_diag + 1e-6)
                    if_var = compute_influence_variance(jac, h_inv, grad_cov_diag)
                    if_vars.append(if_var)

                if_penalty = torch.stack(if_vars).mean()

                loss = sup_loss + lam * if_penalty
            else:
                loss = sup_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(test_x)
        rmse = torch.sqrt(F.mse_loss(preds, test_y)).item()
    return preds.numpy(), rmse


def train_ifr_classification(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    lam=None,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
    fisher_ema=None,
    n_jacobian_samples=None,
):
    """
    Influence Function Regularization for classification.

    FIXES:
    1. Uses torch.autograd.grad to avoid gradient pollution
    2. Correct influence variance formula
    3. Reasonable Fisher initialization
    """
    cfg_cls = CLASSIFICATION_CONFIG
    cfg_ifr = IFR_CONFIG
    lam = lam if lam is not None else cfg_ifr["lam"]
    fisher_ema = fisher_ema if fisher_ema is not None else cfg_ifr["fisher_ema"]
    n_jacobian_samples = (
        n_jacobian_samples
        if n_jacobian_samples is not None
        else cfg_ifr["n_jacobian_samples"]
    )
    epochs = epochs or cfg_cls["epochs"]
    bs = bs or cfg_cls["batch_size"]
    lr = lr or cfg_cls["lr"]
    hid = hid or cfg_cls["hidden_dim"]
    dropout = dropout or cfg_cls["dropout"]

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    n_params = sum(p.numel() for p in model.parameters())
    fisher_diag = torch.ones(n_params) * cfg_ifr["fisher_init"]

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            B = len(xb)

            logits = model(xb)
            sup_loss = F.cross_entropy(logits, yb)

            if epoch >= 2 and lam > 0:
                per_sample_grads = compute_per_sample_gradients(
                    model, xb, yb, lambda p, y: F.cross_entropy(p, y)
                )

                grad_cov_diag = per_sample_grads.var(dim=0)
                fisher_diag = (
                    fisher_ema * fisher_diag
                    + (1 - fisher_ema) * (per_sample_grads**2).mean(dim=0)
                )

                sample_idx = torch.randperm(B)[: min(n_jacobian_samples, B)]
                if_vars = []
                for idx in sample_idx:
                    jac = compute_prediction_jacobian(model, xb[idx], out_dim=2)
                    jac = jac.detach()
                    h_inv = 1.0 / (fisher_diag + 1e-6)
                    if_var = compute_influence_variance(jac, h_inv, grad_cov_diag)
                    if_vars.append(if_var)

                if_penalty = torch.stack(if_vars).mean()

                loss = sup_loss + lam * if_penalty
            else:
                loss = sup_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        acc = (logits.argmax(1) == test_y).float().mean().item()
    return logits.numpy(), acc
