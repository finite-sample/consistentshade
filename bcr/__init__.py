"""Bootstrap Consistency Regularization (BCR) for stable neural network predictions.

BCR is a PyTorch-native regularization method that improves prediction stability
by training an ensemble of models with bootstrap consistency constraints.

Example usage:

    # High-level API
    from bcr import BCRTrainer
    trainer = BCRTrainer(d_in=10, task='regression', K=3, lam=0.05)
    trainer.fit(train_dataset, epochs=25)
    preds = trainer.predict(test_x)

    # Low-level regularizer for custom training loops
    from bcr import BCRRegularizer
    bcr = BCRRegularizer(model_factory, K=3, lam=0.05)
    loss, metrics = bcr.compute_loss(x, y, loss_fn)
"""

from ._version import __version__
from .config import (
    BCR_CONFIG,
    CLASSIFICATION_CONFIG,
    EXPERIMENT_CONFIG,
    IFR_CONFIG,
    IFR_KFAC_CONFIG,
    RDROP_CONFIG,
    REGRESSION_CONFIG,
    SAM_CONFIG,
    WD_CONFIG,
    BCRConfig,
    ClassificationConfig,
    ExperimentConfig,
    IFRConfig,
    IFRKFACConfig,
    RDropConfig,
    RegressionConfig,
    SAMConfig,
    WeightDecayConfig,
)
from .metrics import (
    MethodComparison,
    StabilityEstimate,
    classification_stability_analysis,
    cohens_d,
    compare_methods_bootstrap,
    comprehensive_stability_metrics,
    logit_stability_rmse,
    logit_stability_with_ci,
    stability_rmse,
    stability_rmse_with_ci,
)
from .models import DropMLP
from .optimizers import SAM
from .regularizers import BCRRegularizer, IFRRegularizer, bcr_loss
from .training import (
    BCRTrainer,
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
from .utils import get_device, set_seed

__all__ = [
    "__version__",
    # High-level API
    "BCRTrainer",
    "BCRRegularizer",
    "IFRRegularizer",
    "bcr_loss",
    # Models
    "DropMLP",
    # Optimizers
    "SAM",
    # Training functions
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
    # Metrics
    "stability_rmse",
    "stability_rmse_with_ci",
    "compare_methods_bootstrap",
    "logit_stability_rmse",
    "logit_stability_with_ci",
    "cohens_d",
    "comprehensive_stability_metrics",
    "classification_stability_analysis",
    "StabilityEstimate",
    "MethodComparison",
    # Config
    "BCR_CONFIG",
    "REGRESSION_CONFIG",
    "CLASSIFICATION_CONFIG",
    "SAM_CONFIG",
    "RDROP_CONFIG",
    "WD_CONFIG",
    "IFR_CONFIG",
    "IFR_KFAC_CONFIG",
    "EXPERIMENT_CONFIG",
    "BCRConfig",
    "RegressionConfig",
    "ClassificationConfig",
    "SAMConfig",
    "RDropConfig",
    "WeightDecayConfig",
    "IFRConfig",
    "IFRKFACConfig",
    "ExperimentConfig",
    # Utils
    "set_seed",
    "get_device",
]
