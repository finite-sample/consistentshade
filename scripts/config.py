"""Configuration and hyperparameters for experiments."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABS_DIR = os.path.join(BASE_DIR, "tabs")
FIGS_DIR = os.path.join(BASE_DIR, "figs")

REGRESSION_CONFIG = {
    "epochs": 25,
    "batch_size": 64,
    "lr": 1e-3,
    "hidden_dim": 64,
    "dropout": 0.2,
}

CLASSIFICATION_CONFIG = {
    "epochs": 20,
    "batch_size": 256,
    "lr": 3e-4,
    "hidden_dim": 128,
    "dropout": 0.3,
}

BCR_CONFIG = {
    "K": 3,
    "lam": 0.05,
}

SAM_CONFIG = {
    "rho": 0.05,
}

RDROP_CONFIG = {
    "alpha": 0.5,
}

WD_CONFIG = {
    "weight_decay": 1e-2,
}

IFR_CONFIG = {
    "lam": 0.1,
    "fisher_ema": 0.99,
    "n_jacobian_samples": 8,
    "fisher_init": 1e-2,
}

EXPERIMENT_CONFIG = {
    "n_replicates": 30,
    "base_seed": 12345,
}
