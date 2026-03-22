"""Training functions for all methods."""

from .bagging import train_bagging_classification, train_bagging_regression
from .baseline import train_baseline_classification, train_baseline_regression
from .bcr import train_bcr_classification, train_bcr_regression
from .ifr import train_ifr_classification, train_ifr_regression
from .ifr_kfac import train_ifr_kfac_classification, train_ifr_kfac_regression
from .rdrop import train_rdrop_classification, train_rdrop_regression
from .sam import train_sam_classification, train_sam_regression
from .weight_decay import train_wd_classification, train_wd_regression

__all__ = [
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
