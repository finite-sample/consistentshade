"""R-Drop regularization training."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from ..config import CLASSIFICATION_CONFIG, RDROP_CONFIG, REGRESSION_CONFIG
from ..models import DropMLP
from ..utils import set_seed


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
    alpha = alpha if alpha is not None else cfg_rdrop["alpha"]
    epochs = epochs or cfg_reg["epochs"]
    bs = bs or cfg_reg["batch_size"]
    lr = lr or cfg_reg["lr"]
    hid = hid or cfg_reg["hidden_dim"]
    dropout = dropout or cfg_reg["dropout"]

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
    alpha = alpha if alpha is not None else cfg_rdrop["alpha"]
    epochs = epochs or cfg_cls["epochs"]
    bs = bs or cfg_cls["batch_size"]
    lr = lr or cfg_cls["lr"]
    hid = hid or cfg_cls["hidden_dim"]
    dropout = dropout or cfg_cls["dropout"]

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
