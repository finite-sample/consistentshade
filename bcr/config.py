"""Configuration dataclasses for BCR experiments."""

from dataclasses import dataclass


@dataclass
class RegressionConfig:
    """Default configuration for regression tasks."""

    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    hidden_dim: int = 64
    dropout: float = 0.2


@dataclass
class ClassificationConfig:
    """Default configuration for classification tasks."""

    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4
    hidden_dim: int = 128
    dropout: float = 0.3


@dataclass
class BCRConfig:
    """Configuration for Bootstrap Consistency Regularization."""

    K: int = 3
    lam: float = 0.05


@dataclass
class SAMConfig:
    """Configuration for Sharpness-Aware Minimization."""

    rho: float = 0.05


@dataclass
class RDropConfig:
    """Configuration for R-Drop regularization."""

    alpha: float = 0.5


@dataclass
class WeightDecayConfig:
    """Configuration for weight decay."""

    weight_decay: float = 1e-2


@dataclass
class IFRConfig:
    """Configuration for Influence Function Regularization."""

    lam: float = 0.1
    fisher_ema: float = 0.99
    n_jacobian_samples: int = 8
    fisher_init: float = 1e-2


@dataclass
class IFRKFACConfig:
    """Configuration for KFAC-based IFR."""

    lam: float = 0.1
    kfac_ema: float = 0.95
    kfac_damping: float = 1e-2
    n_jacobian_samples: int = 16
    warmup_epochs: int = 3


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""

    n_replicates: int = 30
    base_seed: int = 12345


REGRESSION_CONFIG = RegressionConfig()
CLASSIFICATION_CONFIG = ClassificationConfig()
BCR_CONFIG = BCRConfig()
SAM_CONFIG = SAMConfig()
RDROP_CONFIG = RDropConfig()
WD_CONFIG = WeightDecayConfig()
IFR_CONFIG = IFRConfig()
IFR_KFAC_CONFIG = IFRKFACConfig()
EXPERIMENT_CONFIG = ExperimentConfig()
