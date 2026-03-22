"""KFAC-based Influence Function Regularization training.

This module implements IFR using Kronecker-Factored Approximate Curvature (KFAC)
for better Hessian approximation than diagonal Fisher.

KFAC approximates the Fisher for each layer as:
    F_l = A_l ⊗ G_l
    F_l^{-1} = A_l^{-1} ⊗ G_l^{-1}

where A_l is the input activation covariance and G_l is the output gradient covariance.

Reference: Martens & Grosse (2015) "Optimizing Neural Networks with Kronecker-factored
Approximate Curvature"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from ..config import CLASSIFICATION_CONFIG, IFR_KFAC_CONFIG, REGRESSION_CONFIG
from ..models import DropMLP
from ..utils import set_seed


class KFACState:
    """Maintains running KFAC factors A (input cov) and G (output grad cov) per layer."""

    def __init__(self, model, ema=0.95, damping=1e-2):
        self.ema = ema
        self.damping = damping
        self.A = {}
        self.G = {}
        self.layer_names = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.layer_names.append(name)
                in_features = module.in_features
                out_features = module.out_features
                self.A[name] = torch.eye(in_features + 1) * damping
                self.G[name] = torch.eye(out_features) * damping

    def update(self, name, a, g):
        """
        Update KFAC factors for a layer.

        Args:
            name: Layer name
            a: Input activations [B, in_features] (will append 1s for bias)
            g: Output gradients [B, out_features]
        """
        B = a.shape[0]

        a_with_bias = torch.cat([a, torch.ones(B, 1, device=a.device)], dim=1)
        A_batch = (a_with_bias.T @ a_with_bias) / B
        G_batch = (g.T @ g) / B

        self.A[name] = self.ema * self.A[name] + (1 - self.ema) * A_batch.detach()
        self.G[name] = self.ema * self.G[name] + (1 - self.ema) * G_batch.detach()

    def compute_layer_h_inv(self, name):
        """Compute damped inverse of KFAC factors for a layer."""
        A = self.A[name]
        G = self.G[name]

        A_damped = A + self.damping * torch.eye(A.shape[0], device=A.device)
        G_damped = G + self.damping * torch.eye(G.shape[0], device=G.device)

        try:
            A_inv = torch.linalg.inv(A_damped)
            G_inv = torch.linalg.inv(G_damped)
        except RuntimeError:
            A_inv = torch.linalg.pinv(A_damped)
            G_inv = torch.linalg.pinv(G_damped)

        return A_inv, G_inv


def compute_per_sample_gradients_vmap(model, xb, yb, loss_fn):
    """
    Compute per-sample gradients using torch.func.vmap for efficiency.

    This is 10-50x faster than sequential loop for moderate batch sizes.
    """
    params = dict(model.named_parameters())
    param_names = list(params.keys())
    param_vals = tuple(params.values())

    def compute_loss(param_vals, x, y):
        param_dict = dict(zip(param_names, param_vals))
        pred = torch.func.functional_call(model, param_dict, x.unsqueeze(0))
        return loss_fn(pred, y.unsqueeze(0))

    grad_fn = torch.func.grad(compute_loss)
    vmapped_grad_fn = torch.func.vmap(grad_fn, in_dims=(None, 0, 0))

    per_sample_grads = vmapped_grad_fn(param_vals, xb, yb)

    grads_flat = []
    for i in range(len(xb)):
        grad_i = torch.cat([per_sample_grads[j][i].flatten() for j in range(len(param_vals))])
        grads_flat.append(grad_i)

    return torch.stack(grads_flat)


def compute_layerwise_jacobian(model, x, out_dim=1):
    """
    Compute Jacobian of output w.r.t. each layer's parameters.

    Returns dict mapping layer_name -> Jacobian tensor
    """
    jac_dict = {}
    pred = model(x.unsqueeze(0))

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight
            bias = module.bias

            jac_w = []
            jac_b = []

            for k in range(out_dim):
                if out_dim == 1:
                    target = pred.sum()
                else:
                    target = pred[0, k]

                grad_w = torch.autograd.grad(
                    target, weight, retain_graph=True, create_graph=False
                )[0]
                grad_b = torch.autograd.grad(
                    target, bias, retain_graph=True, create_graph=False
                )[0]

                jac_w.append(grad_w.flatten())
                jac_b.append(grad_b.flatten())

            jac_w = torch.stack(jac_w) if out_dim > 1 else jac_w[0].unsqueeze(0)
            jac_b = torch.stack(jac_b) if out_dim > 1 else jac_b[0].unsqueeze(0)

            jac_dict[name] = (jac_w, jac_b)

    return jac_dict


def compute_influence_variance_kfac(jac_dict, kfac_state, grad_cov_dict):
    """
    Compute influence-based prediction variance using KFAC approximation.

    σ²(x) = J^T H^{-1} Σ_g H^{-1} J

    For KFAC, we have per-layer:
        H_l^{-1} = A_l^{-1} ⊗ G_l^{-1}

    The quadratic form becomes:
        J_l^T (A_l^{-1} ⊗ G_l^{-1}) Σ_{g,l} (A_l^{-1} ⊗ G_l^{-1}) J_l

    This can be computed efficiently using the Kronecker product identity:
        vec(X)^T (A ⊗ B) vec(Y) = tr(X^T A Y B^T)
    """
    total_var = 0.0

    for name in kfac_state.layer_names:
        if name not in jac_dict:
            continue

        A_inv, G_inv = kfac_state.compute_layer_h_inv(name)
        jac_w, jac_b = jac_dict[name]

        out_dim = jac_w.shape[0] if jac_w.dim() > 1 else 1
        out_features = G_inv.shape[0]
        in_features = A_inv.shape[0] - 1

        for k in range(out_dim):
            jac_w_k = jac_w[k] if out_dim > 1 else jac_w.squeeze(0)
            jac_b_k = jac_b[k] if out_dim > 1 else jac_b.squeeze(0)

            J_mat = jac_w_k.reshape(out_features, in_features)
            J_bias = jac_b_k

            grad_cov_w, grad_cov_b = grad_cov_dict.get(name, (None, None))

            if grad_cov_w is None:
                v_w = (J_mat @ A_inv[:in_features, :in_features] @ J_mat.T).trace()
                v_w = (G_inv @ torch.eye(out_features) * v_w @ G_inv.T).trace()
            else:
                HinvJ_w = G_inv @ J_mat @ A_inv[:in_features, :in_features]
                cov_reshaped = grad_cov_w.reshape(out_features, in_features)
                v_w = (HinvJ_w * cov_reshaped * HinvJ_w).sum()

            v_b = (jac_b_k * A_inv[-1, -1] * G_inv.diag() ** 2).sum()

            total_var = total_var + v_w + v_b

    return total_var


def compute_grad_covariance_layerwise(model, xb, yb, loss_fn):
    """Compute per-layer gradient covariance for KFAC influence computation."""
    grad_cov_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            grads_w = []
            grads_b = []

            for i in range(len(xb)):
                model.zero_grad()
                pred = model(xb[i : i + 1])
                loss = loss_fn(pred, yb[i : i + 1])
                loss.backward(retain_graph=False)

                grads_w.append(module.weight.grad.flatten().clone())
                grads_b.append(module.bias.grad.flatten().clone())

            grads_w = torch.stack(grads_w)
            grads_b = torch.stack(grads_b)

            grad_cov_dict[name] = (grads_w.var(dim=0), grads_b.var(dim=0))

    return grad_cov_dict


class KFACHooks:
    """Forward hooks to capture activations for KFAC updates."""

    def __init__(self, model):
        self.activations = {}
        self.hooks = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = input[0].detach()

        return hook

    def clear(self):
        self.activations = {}

    def remove(self):
        for hook in self.hooks:
            hook.remove()


def train_ifr_kfac_regression(
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
    kfac_ema=None,
    kfac_damping=None,
    n_jacobian_samples=None,
    warmup_epochs=None,
):
    """
    KFAC-based Influence Function Regularization for regression.

    Uses Kronecker-Factored Approximate Curvature for better Hessian approximation
    than diagonal Fisher.
    """
    cfg_reg = REGRESSION_CONFIG
    cfg_ifr = IFR_KFAC_CONFIG
    lam = lam if lam is not None else cfg_ifr["lam"]
    kfac_ema = kfac_ema if kfac_ema is not None else cfg_ifr["kfac_ema"]
    kfac_damping = kfac_damping if kfac_damping is not None else cfg_ifr["kfac_damping"]
    n_jacobian_samples = (
        n_jacobian_samples
        if n_jacobian_samples is not None
        else cfg_ifr["n_jacobian_samples"]
    )
    warmup_epochs = warmup_epochs if warmup_epochs is not None else cfg_ifr["warmup_epochs"]
    epochs = epochs or cfg_reg["epochs"]
    bs = bs or cfg_reg["batch_size"]
    lr = lr or cfg_reg["lr"]
    hid = hid or cfg_reg["hidden_dim"]
    dropout = dropout or cfg_reg["dropout"]

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    kfac_state = KFACState(model, ema=kfac_ema, damping=kfac_damping)
    hooks = KFACHooks(model)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            B = len(xb)
            hooks.clear()

            preds = model(xb)
            sup_loss = F.mse_loss(preds, yb)

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and name in hooks.activations:
                    a = hooks.activations[name]
                    model.zero_grad()
                    dummy_loss = (preds**2).mean()
                    dummy_loss.backward(retain_graph=True)
                    if module.weight.grad is not None:
                        g = module.weight.grad.T
                        if g.shape[0] == B:
                            kfac_state.update(name, a, g)
                        else:
                            g_repeated = g.unsqueeze(0).expand(B, -1, -1).mean(dim=2)
                            if g_repeated.shape[1] == module.out_features:
                                kfac_state.update(name, a, g_repeated)

            if epoch >= warmup_epochs and lam > 0:
                grad_cov_dict = {}
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        grads_w = []
                        grads_b = []
                        for i in range(min(B, 16)):
                            model.zero_grad()
                            pred_i = model(xb[i : i + 1])
                            loss_i = F.mse_loss(pred_i, yb[i : i + 1])
                            loss_i.backward(retain_graph=False)
                            grads_w.append(module.weight.grad.flatten().detach().clone())
                            grads_b.append(module.bias.grad.flatten().detach().clone())
                        grads_w = torch.stack(grads_w)
                        grads_b = torch.stack(grads_b)
                        grad_cov_dict[name] = (grads_w.var(dim=0), grads_b.var(dim=0))

                sample_idx = torch.randperm(B)[: min(n_jacobian_samples, B)]
                if_vars = []
                for idx in sample_idx:
                    model.zero_grad()
                    jac_dict = compute_layerwise_jacobian(model, xb[idx], out_dim=1)
                    if_var = compute_influence_variance_kfac(jac_dict, kfac_state, grad_cov_dict)
                    if_vars.append(if_var)

                if if_vars:
                    if_penalty = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in if_vars]).mean()
                    loss = sup_loss + lam * if_penalty
                else:
                    loss = sup_loss
            else:
                loss = sup_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

    hooks.remove()
    model.eval()
    with torch.no_grad():
        preds = model(test_x)
        rmse = torch.sqrt(F.mse_loss(preds, test_y)).item()
    return preds.numpy(), rmse


def train_ifr_kfac_classification(
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
    kfac_ema=None,
    kfac_damping=None,
    n_jacobian_samples=None,
    warmup_epochs=None,
):
    """
    KFAC-based Influence Function Regularization for classification.

    Uses Kronecker-Factored Approximate Curvature for better Hessian approximation.
    """
    cfg_cls = CLASSIFICATION_CONFIG
    cfg_ifr = IFR_KFAC_CONFIG
    lam = lam if lam is not None else cfg_ifr["lam"]
    kfac_ema = kfac_ema if kfac_ema is not None else cfg_ifr["kfac_ema"]
    kfac_damping = kfac_damping if kfac_damping is not None else cfg_ifr["kfac_damping"]
    n_jacobian_samples = (
        n_jacobian_samples
        if n_jacobian_samples is not None
        else cfg_ifr["n_jacobian_samples"]
    )
    warmup_epochs = warmup_epochs if warmup_epochs is not None else cfg_ifr["warmup_epochs"]
    epochs = epochs or cfg_cls["epochs"]
    bs = bs or cfg_cls["batch_size"]
    lr = lr or cfg_cls["lr"]
    hid = hid or cfg_cls["hidden_dim"]
    dropout = dropout or cfg_cls["dropout"]

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    kfac_state = KFACState(model, ema=kfac_ema, damping=kfac_damping)
    hooks = KFACHooks(model)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            B = len(xb)
            hooks.clear()

            logits = model(xb)
            sup_loss = F.cross_entropy(logits, yb)

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and name in hooks.activations:
                    a = hooks.activations[name]
                    model.zero_grad()
                    dummy_loss = (logits**2).mean()
                    dummy_loss.backward(retain_graph=True)
                    if module.weight.grad is not None:
                        g = module.weight.grad.T
                        if g.shape[0] == B:
                            kfac_state.update(name, a, g)

            if epoch >= warmup_epochs and lam > 0:
                grad_cov_dict = {}
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        grads_w = []
                        grads_b = []
                        for i in range(min(B, 16)):
                            model.zero_grad()
                            logit_i = model(xb[i : i + 1])
                            loss_i = F.cross_entropy(logit_i, yb[i : i + 1])
                            loss_i.backward(retain_graph=False)
                            grads_w.append(module.weight.grad.flatten().detach().clone())
                            grads_b.append(module.bias.grad.flatten().detach().clone())
                        grads_w = torch.stack(grads_w)
                        grads_b = torch.stack(grads_b)
                        grad_cov_dict[name] = (grads_w.var(dim=0), grads_b.var(dim=0))

                sample_idx = torch.randperm(B)[: min(n_jacobian_samples, B)]
                if_vars = []
                for idx in sample_idx:
                    model.zero_grad()
                    jac_dict = compute_layerwise_jacobian(model, xb[idx], out_dim=2)
                    if_var = compute_influence_variance_kfac(jac_dict, kfac_state, grad_cov_dict)
                    if_vars.append(if_var)

                if if_vars:
                    if_penalty = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in if_vars]).mean()
                    loss = sup_loss + lam * if_penalty
                else:
                    loss = sup_loss
            else:
                loss = sup_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

    hooks.remove()
    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        acc = (logits.argmax(1) == test_y).float().mean().item()
    return logits.numpy(), acc
