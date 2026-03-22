"""Standard bagging training."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from ..config import BCR_CONFIG, CLASSIFICATION_CONFIG, REGRESSION_CONFIG
from ..models import DropMLP
from ..utils import set_seed


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
    K = K if K is not None else cfg_bcr["K"]
    epochs = epochs or cfg_reg["epochs"]
    bs = bs or cfg_reg["batch_size"]
    lr = lr or cfg_reg["lr"]
    hid = hid or cfg_reg["hidden_dim"]
    dropout = dropout or cfg_reg["dropout"]

    set_seed(seed)
    models = []

    X_train = train_ds.tensors[0].numpy()
    y_train = train_ds.tensors[1].numpy()
    n = len(X_train)

    for k in range(K):
        set_seed(seed + k * 1000)
        boot_idx = np.random.choice(n, n, replace=True)
        boot_ds = TensorDataset(
            torch.tensor(X_train[boot_idx]), torch.tensor(y_train[boot_idx])
        )

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
    K = K if K is not None else cfg_bcr["K"]
    epochs = epochs or cfg_cls["epochs"]
    bs = bs or cfg_cls["batch_size"]
    lr = lr or cfg_cls["lr"]
    hid = hid or cfg_cls["hidden_dim"]
    dropout = dropout or cfg_cls["dropout"]

    set_seed(seed)
    models = []

    X_train = train_ds.tensors[0].numpy()
    y_train = train_ds.tensors[1].numpy()
    n = len(X_train)

    for k in range(K):
        set_seed(seed + k * 1000)
        boot_idx = np.random.choice(n, n, replace=True)
        boot_ds = TensorDataset(
            torch.tensor(X_train[boot_idx]), torch.tensor(y_train[boot_idx])
        )

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
