"""Baseline ERM training."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from ..config import CLASSIFICATION_CONFIG, REGRESSION_CONFIG
from ..models import DropMLP
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
    """
    Train baseline MLP for regression.

    FIX: Uses single forward pass in eval mode (not fake MC dropout).
    The original code did 4 forward passes with model.eval() which
    produces identical results due to disabled dropout.
    """
    cfg = REGRESSION_CONFIG
    epochs = epochs or cfg["epochs"]
    bs = bs or cfg["batch_size"]
    lr = lr or cfg["lr"]
    hid = hid or cfg["hidden_dim"]
    dropout = dropout or cfg["dropout"]

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
    """
    Train baseline MLP for classification.

    FIX: Uses single forward pass in eval mode (not fake MC dropout).
    """
    cfg = CLASSIFICATION_CONFIG
    epochs = epochs or cfg["epochs"]
    bs = bs or cfg["batch_size"]
    lr = lr or cfg["lr"]
    hid = hid or cfg["hidden_dim"]
    dropout = dropout or cfg["dropout"]

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
