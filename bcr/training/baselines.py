"""Baseline training methods for comparison."""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from ..config import (
    BCR_CONFIG,
    CLASSIFICATION_CONFIG,
    IFR_CONFIG,
    IFR_KFAC_CONFIG,
    RDROP_CONFIG,
    REGRESSION_CONFIG,
    SAM_CONFIG,
    WD_CONFIG,
)
from ..models import DropMLP
from ..optimizers import SAM
from ..regularizers.ifr import compute_per_sample_gradients, compute_prediction_jacobian, compute_influence_variance
from ..regularizers.kfac import (
    KFACHooks,
    KFACState,
    compute_influence_variance_kfac,
    compute_layerwise_jacobian,
)
from ..utils import set_seed


def train_baseline_regression(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
):
    """Train baseline MLP for regression."""
    cfg = REGRESSION_CONFIG
    epochs = epochs or cfg.epochs
    bs = bs or cfg.batch_size
    lr = lr or cfg.lr
    hid = hid or cfg.hidden_dim
    dropout = dropout or cfg.dropout

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(test_x)
        rmse = torch.sqrt(F.mse_loss(preds, test_y)).item()
    return preds.numpy(), rmse


def train_baseline_classification(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
):
    """Train baseline MLP for classification."""
    cfg = CLASSIFICATION_CONFIG
    epochs = epochs or cfg.epochs
    bs = bs or cfg.batch_size
    lr = lr or cfg.lr
    hid = hid or cfg.hidden_dim
    dropout = dropout or cfg.dropout

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        acc = (logits.argmax(1) == test_y).float().mean().item()
    return logits.numpy(), acc


def train_bcr_regression(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    K=None,
    lam=None,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
):
    """Bootstrap Consistency Regularization for regression."""
    cfg_reg = REGRESSION_CONFIG
    cfg_bcr = BCR_CONFIG
    K = K if K is not None else cfg_bcr.K
    lam = lam if lam is not None else cfg_bcr.lam
    epochs = epochs or cfg_reg.epochs
    bs = bs or cfg_reg.batch_size
    lr = lr or cfg_reg.lr
    hid = hid or cfg_reg.hidden_dim
    dropout = dropout or cfg_reg.dropout

    set_seed(seed)
    models = nn.ModuleList([DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=1) for _ in range(K)])
    opt = torch.optim.Adam(itertools.chain(*(m.parameters() for m in models)), lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    for _ in range(epochs):
        for xb, yb in loader:
            B = len(xb)
            preds = []
            sup_losses = []

            for m in models:
                px = m(xb)
                preds.append(px)
                weights = torch.poisson(torch.ones(B))
                weighted_loss = (weights * (px - yb) ** 2).mean()
                sup_losses.append(weighted_loss)

            preds = torch.stack(preds)
            sup_loss = torch.stack(sup_losses).mean()
            var_pen = preds.var(dim=0, unbiased=False).mean()
            loss = sup_loss + lam * var_pen

            opt.zero_grad()
            loss.backward()
            opt.step()

    for m in models:
        m.eval()
    with torch.no_grad():
        ens_preds = torch.stack([m(test_x) for m in models]).mean(0)
        rmse = torch.sqrt(F.mse_loss(ens_preds, test_y)).item()
    return ens_preds.numpy(), rmse


def train_bcr_classification(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    K=None,
    lam=None,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
):
    """Bootstrap Consistency Regularization for classification."""
    cfg_cls = CLASSIFICATION_CONFIG
    cfg_bcr = BCR_CONFIG
    K = K if K is not None else cfg_bcr.K
    lam = lam if lam is not None else cfg_bcr.lam
    epochs = epochs or cfg_cls.epochs
    bs = bs or cfg_cls.batch_size
    lr = lr or cfg_cls.lr
    hid = hid or cfg_cls.hidden_dim
    dropout = dropout or cfg_cls.dropout

    set_seed(seed)
    models = nn.ModuleList([DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=2) for _ in range(K)])
    opt = torch.optim.Adam(itertools.chain(*(m.parameters() for m in models)), lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    for _ in range(epochs):
        for xb, yb in loader:
            B = len(xb)
            logits_list = []
            probs_list = []
            sup_losses = []

            for m in models:
                out = m(xb)
                logits_list.append(out)
                probs_list.append(F.softmax(out, dim=-1))
                weights = torch.poisson(torch.ones(B))
                ce_per_sample = F.cross_entropy(out, yb, reduction="none")
                weighted_loss = (weights * ce_per_sample).mean()
                sup_losses.append(weighted_loss)

            probs_stack = torch.stack(probs_list)
            sup_loss = torch.stack(sup_losses).mean()
            var_pen = probs_stack.var(dim=0).mean()
            loss = sup_loss + lam * var_pen

            opt.zero_grad()
            loss.backward()
            opt.step()

    for m in models:
        m.eval()
    with torch.no_grad():
        logits = torch.stack([m(test_x) for m in models]).mean(0)
        acc = (logits.argmax(1) == test_y).float().mean().item()
    return logits.numpy(), acc


def train_wd_regression(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    weight_decay=None,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
):
    """Train MLP with weight decay for regression."""
    cfg_reg = REGRESSION_CONFIG
    cfg_wd = WD_CONFIG
    weight_decay = weight_decay if weight_decay is not None else cfg_wd.weight_decay
    epochs = epochs or cfg_reg.epochs
    bs = bs or cfg_reg.batch_size
    lr = lr or cfg_reg.lr
    hid = hid or cfg_reg.hidden_dim
    dropout = dropout or cfg_reg.dropout

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(test_x)
        rmse = torch.sqrt(F.mse_loss(preds, test_y)).item()
    return preds.numpy(), rmse


def train_wd_classification(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    weight_decay=None,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
):
    """Train MLP with weight decay for classification."""
    cfg_cls = CLASSIFICATION_CONFIG
    cfg_wd = WD_CONFIG
    weight_decay = weight_decay if weight_decay is not None else cfg_wd.weight_decay
    epochs = epochs or cfg_cls.epochs
    bs = bs or cfg_cls.batch_size
    lr = lr or cfg_cls.lr
    hid = hid or cfg_cls.hidden_dim
    dropout = dropout or cfg_cls.dropout

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        acc = (logits.argmax(1) == test_y).float().mean().item()
    return logits.numpy(), acc


def train_sam_regression(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    rho=None,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
):
    """Train MLP with SAM optimizer for regression."""
    cfg_reg = REGRESSION_CONFIG
    cfg_sam = SAM_CONFIG
    rho = rho if rho is not None else cfg_sam.rho
    epochs = epochs or cfg_reg.epochs
    bs = bs or cfg_reg.batch_size
    lr = lr or cfg_reg.lr
    hid = hid or cfg_reg.hidden_dim
    dropout = dropout or cfg_reg.dropout

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=1)
    opt = SAM(model.parameters(), torch.optim.Adam, rho=rho, lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.first_step()

            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            opt.second_step()

    model.eval()
    with torch.no_grad():
        preds = model(test_x)
        rmse = torch.sqrt(F.mse_loss(preds, test_y)).item()
    return preds.numpy(), rmse


def train_sam_classification(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    rho=None,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
):
    """Train MLP with SAM optimizer for classification."""
    cfg_cls = CLASSIFICATION_CONFIG
    cfg_sam = SAM_CONFIG
    rho = rho if rho is not None else cfg_sam.rho
    epochs = epochs or cfg_cls.epochs
    bs = bs or cfg_cls.batch_size
    lr = lr or cfg_cls.lr
    hid = hid or cfg_cls.hidden_dim
    dropout = dropout or cfg_cls.dropout

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=2)
    opt = SAM(model.parameters(), torch.optim.Adam, rho=rho, lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.first_step()

            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.second_step()

    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        acc = (logits.argmax(1) == test_y).float().mean().item()
    return logits.numpy(), acc


def kl_div_regression(p1, p2, sigma=1.0):
    """Symmetric KL-like divergence for regression (Gaussian assumption)."""
    return ((p1 - p2) ** 2).mean() / (2 * sigma**2)


def train_rdrop_regression(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    alpha=None,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
):
    """Train MLP with R-Drop for regression."""
    cfg_reg = REGRESSION_CONFIG
    cfg_rdrop = RDROP_CONFIG
    alpha = alpha if alpha is not None else cfg_rdrop.alpha
    epochs = epochs or cfg_reg.epochs
    bs = bs or cfg_reg.batch_size
    lr = lr or cfg_reg.lr
    hid = hid or cfg_reg.hidden_dim
    dropout = dropout or cfg_reg.dropout

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            pred1 = model(xb)
            pred2 = model(xb)

            loss1 = F.mse_loss(pred1, yb)
            loss2 = F.mse_loss(pred2, yb)
            kl_loss = kl_div_regression(pred1, pred2)

            loss = (loss1 + loss2) / 2 + alpha * kl_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(test_x)
        rmse = torch.sqrt(F.mse_loss(preds, test_y)).item()
    return preds.numpy(), rmse


def train_rdrop_classification(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    alpha=None,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
):
    """Train MLP with R-Drop for classification."""
    cfg_cls = CLASSIFICATION_CONFIG
    cfg_rdrop = RDROP_CONFIG
    alpha = alpha if alpha is not None else cfg_rdrop.alpha
    epochs = epochs or cfg_cls.epochs
    bs = bs or cfg_cls.batch_size
    lr = lr or cfg_cls.lr
    hid = hid or cfg_cls.hidden_dim
    dropout = dropout or cfg_cls.dropout

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            logits1 = model(xb)
            logits2 = model(xb)

            loss1 = F.cross_entropy(logits1, yb)
            loss2 = F.cross_entropy(logits2, yb)

            p1 = F.softmax(logits1, dim=-1)
            p2 = F.softmax(logits2, dim=-1)
            kl_loss = (
                F.kl_div(p1.log(), p2, reduction="batchmean")
                + F.kl_div(p2.log(), p1, reduction="batchmean")
            ) / 2

            loss = (loss1 + loss2) / 2 + alpha * kl_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        acc = (logits.argmax(1) == test_y).float().mean().item()
    return logits.numpy(), acc


def train_bagging_regression(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    K=None,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
):
    """Standard bagging: K independent models trained on bootstrap samples."""
    cfg_reg = REGRESSION_CONFIG
    cfg_bcr = BCR_CONFIG
    K = K if K is not None else cfg_bcr.K
    epochs = epochs or cfg_reg.epochs
    bs = bs or cfg_reg.batch_size
    lr = lr or cfg_reg.lr
    hid = hid or cfg_reg.hidden_dim
    dropout = dropout or cfg_reg.dropout

    set_seed(seed)
    models = []

    X_train = train_ds.tensors[0].numpy()
    y_train = train_ds.tensors[1].numpy()
    n = len(X_train)

    for k in range(K):
        set_seed(seed + k * 1000)
        boot_idx = np.random.choice(n, n, replace=True)
        boot_ds = TensorDataset(torch.tensor(X_train[boot_idx]), torch.tensor(y_train[boot_idx]))

        model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=1)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loader = DataLoader(boot_ds, batch_size=bs, sampler=RandomSampler(boot_ds))

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                loss = F.mse_loss(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        models.append(model)

    for m in models:
        m.eval()
    with torch.no_grad():
        ens_preds = torch.stack([m(test_x) for m in models]).mean(0)
        rmse = torch.sqrt(F.mse_loss(ens_preds, test_y)).item()
    return ens_preds.numpy(), rmse


def train_bagging_classification(
    seed,
    train_ds,
    test_x,
    test_y,
    d_in,
    K=None,
    epochs=None,
    bs=None,
    lr=None,
    hid=None,
    dropout=None,
):
    """Standard bagging: K independent models trained on bootstrap samples."""
    cfg_cls = CLASSIFICATION_CONFIG
    cfg_bcr = BCR_CONFIG
    K = K if K is not None else cfg_bcr.K
    epochs = epochs or cfg_cls.epochs
    bs = bs or cfg_cls.batch_size
    lr = lr or cfg_cls.lr
    hid = hid or cfg_cls.hidden_dim
    dropout = dropout or cfg_cls.dropout

    set_seed(seed)
    models = []

    X_train = train_ds.tensors[0].numpy()
    y_train = train_ds.tensors[1].numpy()
    n = len(X_train)

    for k in range(K):
        set_seed(seed + k * 1000)
        boot_idx = np.random.choice(n, n, replace=True)
        boot_ds = TensorDataset(torch.tensor(X_train[boot_idx]), torch.tensor(y_train[boot_idx]))

        model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=2)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loader = DataLoader(boot_ds, batch_size=bs, sampler=RandomSampler(boot_ds))

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        models.append(model)

    for m in models:
        m.eval()
    with torch.no_grad():
        logits = torch.stack([m(test_x) for m in models]).mean(0)
        acc = (logits.argmax(1) == test_y).float().mean().item()
    return logits.numpy(), acc


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
    """Influence Function Regularization for regression."""
    cfg_reg = REGRESSION_CONFIG
    cfg_ifr = IFR_CONFIG
    lam = lam if lam is not None else cfg_ifr.lam
    fisher_ema = fisher_ema if fisher_ema is not None else cfg_ifr.fisher_ema
    n_jacobian_samples = n_jacobian_samples if n_jacobian_samples is not None else cfg_ifr.n_jacobian_samples
    epochs = epochs or cfg_reg.epochs
    bs = bs or cfg_reg.batch_size
    lr = lr or cfg_reg.lr
    hid = hid or cfg_reg.hidden_dim
    dropout = dropout or cfg_reg.dropout

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    n_params = sum(p.numel() for p in model.parameters())
    fisher_diag = torch.ones(n_params) * cfg_ifr.fisher_init

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
                    fisher_ema * fisher_diag + (1 - fisher_ema) * (per_sample_grads**2).mean(dim=0)
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
    """Influence Function Regularization for classification."""
    cfg_cls = CLASSIFICATION_CONFIG
    cfg_ifr = IFR_CONFIG
    lam = lam if lam is not None else cfg_ifr.lam
    fisher_ema = fisher_ema if fisher_ema is not None else cfg_ifr.fisher_ema
    n_jacobian_samples = n_jacobian_samples if n_jacobian_samples is not None else cfg_ifr.n_jacobian_samples
    epochs = epochs or cfg_cls.epochs
    bs = bs or cfg_cls.batch_size
    lr = lr or cfg_cls.lr
    hid = hid or cfg_cls.hidden_dim
    dropout = dropout or cfg_cls.dropout

    set_seed(seed)
    model = DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=bs, sampler=RandomSampler(train_ds))

    n_params = sum(p.numel() for p in model.parameters())
    fisher_diag = torch.ones(n_params) * cfg_ifr.fisher_init

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
                    fisher_ema * fisher_diag + (1 - fisher_ema) * (per_sample_grads**2).mean(dim=0)
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
    """KFAC-based Influence Function Regularization for regression."""
    cfg_reg = REGRESSION_CONFIG
    cfg_ifr = IFR_KFAC_CONFIG
    lam = lam if lam is not None else cfg_ifr.lam
    kfac_ema = kfac_ema if kfac_ema is not None else cfg_ifr.kfac_ema
    kfac_damping = kfac_damping if kfac_damping is not None else cfg_ifr.kfac_damping
    n_jacobian_samples = n_jacobian_samples if n_jacobian_samples is not None else cfg_ifr.n_jacobian_samples
    warmup_epochs = warmup_epochs if warmup_epochs is not None else cfg_ifr.warmup_epochs
    epochs = epochs or cfg_reg.epochs
    bs = bs or cfg_reg.batch_size
    lr = lr or cfg_reg.lr
    hid = hid or cfg_reg.hidden_dim
    dropout = dropout or cfg_reg.dropout

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
                    if_penalty = torch.stack(
                        [v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in if_vars]
                    ).mean()
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
    """KFAC-based Influence Function Regularization for classification."""
    cfg_cls = CLASSIFICATION_CONFIG
    cfg_ifr = IFR_KFAC_CONFIG
    lam = lam if lam is not None else cfg_ifr.lam
    kfac_ema = kfac_ema if kfac_ema is not None else cfg_ifr.kfac_ema
    kfac_damping = kfac_damping if kfac_damping is not None else cfg_ifr.kfac_damping
    n_jacobian_samples = n_jacobian_samples if n_jacobian_samples is not None else cfg_ifr.n_jacobian_samples
    warmup_epochs = warmup_epochs if warmup_epochs is not None else cfg_ifr.warmup_epochs
    epochs = epochs or cfg_cls.epochs
    bs = bs or cfg_cls.batch_size
    lr = lr or cfg_cls.lr
    hid = hid or cfg_cls.hidden_dim
    dropout = dropout or cfg_cls.dropout

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
                    if_penalty = torch.stack(
                        [v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in if_vars]
                    ).mean()
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
