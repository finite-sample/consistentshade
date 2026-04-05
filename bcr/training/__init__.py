"""Training modules for BCR and baseline methods."""

from .baselines import (
    train_bagging_classification,
    train_bagging_regression,
    train_baseline_classification,
    train_baseline_regression,
    train_bcr_classification,
    train_bcr_regression,
    train_ifr_classification,
    train_ifr_kfac_classification,
    train_ifr_kfac_regression,
    train_ifr_regression,
    train_rdrop_classification,
    train_rdrop_regression,
    train_sam_classification,
    train_sam_regression,
    train_wd_classification,
    train_wd_regression,
)
from .trainer import BCRTrainer

__all__ = [
    "BCRTrainer",
    "train_baseline_regression",
    "train_baseline_classification",
    "train_bcr_regression",
    "train_bcr_classification",
    "train_wd_regression",
    "train_wd_classification",
    "train_sam_regression",
    "train_sam_classification",
    "train_rdrop_regression",
    "train_rdrop_classification",
    "train_bagging_regression",
    "train_bagging_classification",
    "train_ifr_regression",
    "train_ifr_classification",
    "train_ifr_kfac_regression",
    "train_ifr_kfac_classification",
]
