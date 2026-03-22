"""Bootstrap Consistency Regularization training."""

import itertools

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from ..config import BCR_CONFIG, CLASSIFICATION_CONFIG, REGRESSION_CONFIG
from ..models import DropMLP
from ..utils import set_seed


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
    """
    Bootstrap Consistency Regularization for regression.

    All K models predict on SAME inputs. Bootstrap via Poisson(1) weighted loss.

    FIX: Weight normalization now uses standard bootstrap weighting.
    Original: weights / weights.sum() changed the bootstrap distribution.
    Fixed: Use (weights * loss).mean() which is standard weighted bootstrap.
    """
    cfg_reg = REGRESSION_CONFIG
    cfg_bcr = BCR_CONFIG
    K = K if K is not None else cfg_bcr["K"]
    lam = lam if lam is not None else cfg_bcr["lam"]
    epochs = epochs or cfg_reg["epochs"]
    bs = bs or cfg_reg["batch_size"]
    lr = lr or cfg_reg["lr"]
    hid = hid or cfg_reg["hidden_dim"]
    dropout = dropout or cfg_reg["dropout"]

    set_seed(seed)
    models = torch.nn.ModuleList(
        [DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=1) for _ in range(K)]
    )
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
    """
    Bootstrap Consistency Regularization for classification.

    FIX: Variance penalty on probabilities instead of raw logits.
    Raw logit variance can be misleading since logits are unbounded.
    """
    cfg_cls = CLASSIFICATION_CONFIG
    cfg_bcr = BCR_CONFIG
    K = K if K is not None else cfg_bcr["K"]
    lam = lam if lam is not None else cfg_bcr["lam"]
    epochs = epochs or cfg_cls["epochs"]
    bs = bs or cfg_cls["batch_size"]
    lr = lr or cfg_cls["lr"]
    hid = hid or cfg_cls["hidden_dim"]
    dropout = dropout or cfg_cls["dropout"]

    set_seed(seed)
    models = torch.nn.ModuleList(
        [DropMLP(d_in=d_in, hid=hid, p=dropout, out_dim=2) for _ in range(K)]
    )
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
