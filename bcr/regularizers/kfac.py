"""KFAC-based Influence Function Regularization utilities.

Implements IFR using Kronecker-Factored Approximate Curvature (KFAC)
for better Hessian approximation than diagonal Fisher.

KFAC approximates the Fisher for each layer as:
    F_l = A_l (x) G_l
    F_l^{-1} = A_l^{-1} (x) G_l^{-1}

where A_l is the input activation covariance and G_l is the output gradient covariance.

Reference: Martens & Grosse (2015) "Optimizing Neural Networks with Kronecker-factored
Approximate Curvature"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KFACState:
    """Maintains running KFAC factors A (input cov) and G (output grad cov) per layer."""

    def __init__(self, model: nn.Module, ema: float = 0.95, damping: float = 1e-2):
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

    def update(self, name: str, a: torch.Tensor, g: torch.Tensor):
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

        self.A[name] = self.ema * self.A[name].to(a.device) + (1 - self.ema) * A_batch.detach()
        self.G[name] = self.ema * self.G[name].to(g.device) + (1 - self.ema) * G_batch.detach()

    def compute_layer_h_inv(self, name: str):
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


class KFACHooks:
    """Forward hooks to capture activations for KFAC updates."""

    def __init__(self, model: nn.Module):
        self.activations = {}
        self.hooks = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name] = input[0].detach()

        return hook

    def clear(self):
        self.activations = {}

    def remove(self):
        for hook in self.hooks:
            hook.remove()


def compute_layerwise_jacobian(model: nn.Module, x: torch.Tensor, out_dim: int = 1):
    """
    Compute Jacobian of output w.r.t. each layer's parameters.

    Returns dict mapping layer_name -> (jac_weight, jac_bias)
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

                grad_w = torch.autograd.grad(target, weight, retain_graph=True, create_graph=False)[0]
                grad_b = torch.autograd.grad(target, bias, retain_graph=True, create_graph=False)[0]

                jac_w.append(grad_w.flatten())
                jac_b.append(grad_b.flatten())

            jac_w = torch.stack(jac_w) if out_dim > 1 else jac_w[0].unsqueeze(0)
            jac_b = torch.stack(jac_b) if out_dim > 1 else jac_b[0].unsqueeze(0)

            jac_dict[name] = (jac_w, jac_b)

    return jac_dict


def compute_influence_variance_kfac(jac_dict: dict, kfac_state: KFACState, grad_cov_dict: dict):
    """
    Compute influence-based prediction variance using KFAC approximation.

    sigma^2(x) = J^T H^{-1} Sigma_g H^{-1} J

    For KFAC, we have per-layer:
        H_l^{-1} = A_l^{-1} (x) G_l^{-1}
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

            grad_cov_w, grad_cov_b = grad_cov_dict.get(name, (None, None))

            if grad_cov_w is None:
                v_w = (J_mat @ A_inv[:in_features, :in_features] @ J_mat.T).trace()
                v_w = (G_inv @ torch.eye(out_features, device=G_inv.device) * v_w @ G_inv.T).trace()
            else:
                HinvJ_w = G_inv @ J_mat @ A_inv[:in_features, :in_features]
                cov_reshaped = grad_cov_w.reshape(out_features, in_features)
                v_w = (HinvJ_w * cov_reshaped * HinvJ_w).sum()

            v_b = (jac_b_k * A_inv[-1, -1] * G_inv.diag() ** 2).sum()

            total_var = total_var + v_w + v_b

    return total_var


def compute_grad_covariance_layerwise(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, loss_fn):
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
