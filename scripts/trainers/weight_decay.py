"""Weight decay training."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from ..config import CLASSIFICATION_CONFIG, REGRESSION_CONFIG, WD_CONFIG
from ..models import DropMLP
from ..utils import set_seed


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
    weight_decay = weight_decay if weight_decay is not None else cfg_wd["weight_decay"]
    epochs = epochs or cfg_reg["epochs"]
    bs = bs or cfg_reg["batch_size"]
    lr = lr or cfg_reg["lr"]
    hid = hid or cfg_reg["hidden_dim"]
    dropout = dropout or cfg_reg["dropout"]

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
    weight_decay = weight_decay if weight_decay is not None else cfg_wd["weight_decay"]
    epochs = epochs or cfg_cls["epochs"]
    bs = bs or cfg_cls["batch_size"]
    lr = lr or cfg_cls["lr"]
    hid = hid or cfg_cls["hidden_dim"]
    dropout = dropout or cfg_cls["dropout"]

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
