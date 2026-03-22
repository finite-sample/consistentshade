"""Stability metrics for regression and classification."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class StabilityEstimate:
    """Container for stability estimate with uncertainty."""
    point: float
    se: float
    ci_lower: float
    ci_upper: float

    def __str__(self):
        return f"{self.point:.4f} (SE={self.se:.4f}, 95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}])"


@dataclass
class MethodComparison:
    """Container for pairwise method comparison."""
    diff: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float

    @property
    def significant_at_05(self) -> bool:
        return self.p_value < 0.05

    @property
    def significant_at_01(self) -> bool:
        return self.p_value < 0.01


def stability_rmse(pred_matrix: np.ndarray) -> float:
    """RMSE of predictions across fits: sqrt(mean variance per sample)."""
    return math.sqrt(pred_matrix.var(axis=0).mean())


def stability_rmse_with_ci(
    pred_matrix: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> StabilityEstimate:
    """
    Compute stability RMSE with bootstrap confidence interval.

    Args:
        pred_matrix: (n_fits, n_samples) array of predictions
        n_bootstrap: number of bootstrap resamples
        ci: confidence level (default 0.95 for 95% CI)
        seed: random seed for reproducibility

    Returns:
        StabilityEstimate with point estimate, SE, and CI bounds
    """
    rng = np.random.RandomState(seed)
    n_fits = pred_matrix.shape[0]

    boot_estimates = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n_fits, n_fits, replace=True)
        resampled = pred_matrix[idx]
        boot_estimates[b] = stability_rmse(resampled)

    point = stability_rmse(pred_matrix)
    se = np.std(boot_estimates, ddof=1)
    alpha = (1 - ci) / 2
    ci_lower = np.percentile(boot_estimates, alpha * 100)
    ci_upper = np.percentile(boot_estimates, (1 - alpha) * 100)

    return StabilityEstimate(
        point=float(point), se=float(se), ci_lower=float(ci_lower), ci_upper=float(ci_upper)
    )


def compare_methods_bootstrap(
    pred_A: np.ndarray,
    pred_B: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> MethodComparison:
    """
    Bootstrap test for difference in stability between two methods.

    Tests H0: stability_A = stability_B

    Args:
        pred_A: (n_fits, n_samples) predictions from method A
        pred_B: (n_fits, n_samples) predictions from method B
        n_bootstrap: number of bootstrap resamples
        ci: confidence level
        seed: random seed

    Returns:
        MethodComparison with difference, SE, CI, and p-value
    """
    rng = np.random.RandomState(seed)
    n_fits = pred_A.shape[0]

    boot_diffs = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n_fits, n_fits, replace=True)
        stab_A = stability_rmse(pred_A[idx])
        stab_B = stability_rmse(pred_B[idx])
        boot_diffs[b] = stab_A - stab_B

    point_diff = stability_rmse(pred_A) - stability_rmse(pred_B)
    se = np.std(boot_diffs, ddof=1)

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(boot_diffs, alpha * 100)
    ci_upper = np.percentile(boot_diffs, (1 - alpha) * 100)

    # Two-sided p-value: proportion of bootstrap samples on opposite side of zero
    p_value = 2 * min(
        np.mean(boot_diffs >= 0),
        np.mean(boot_diffs <= 0)
    )
    p_value = min(p_value, 1.0)

    return MethodComparison(
        diff=float(point_diff),
        se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(p_value),
    )


def logit_stability_rmse(logits_matrix: np.ndarray) -> float:
    """Compute logit stability RMSE for classification."""
    logit_var = logits_matrix.var(axis=0).mean(axis=-1)
    return math.sqrt(logit_var.mean())


def logit_stability_with_ci(
    logits_matrix: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> StabilityEstimate:
    """Compute logit stability RMSE with bootstrap CI."""
    rng = np.random.RandomState(seed)
    n_fits = logits_matrix.shape[0]

    boot_estimates = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n_fits, n_fits, replace=True)
        resampled = logits_matrix[idx]
        boot_estimates[b] = logit_stability_rmse(resampled)

    point = logit_stability_rmse(logits_matrix)
    se = np.std(boot_estimates, ddof=1)
    alpha = (1 - ci) / 2
    ci_lower = np.percentile(boot_estimates, alpha * 100)
    ci_upper = np.percentile(boot_estimates, (1 - alpha) * 100)

    return StabilityEstimate(
        point=float(point), se=float(se), ci_lower=float(ci_lower), ci_upper=float(ci_upper)
    )


def cohens_d(pred_A: np.ndarray, pred_B: np.ndarray, n_bootstrap: int = 500, seed: int = 42) -> float:
    """
    Compute Cohen's d effect size for stability difference.

    Uses pooled SD of bootstrap stability estimates.
    """
    rng = np.random.RandomState(seed)
    n_fits = pred_A.shape[0]

    boot_A = np.zeros(n_bootstrap)
    boot_B = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n_fits, n_fits, replace=True)
        boot_A[b] = stability_rmse(pred_A[idx])
        boot_B[b] = stability_rmse(pred_B[idx])

    mean_A = np.mean(boot_A)
    mean_B = np.mean(boot_B)
    pooled_sd = np.sqrt((np.var(boot_A, ddof=1) + np.var(boot_B, ddof=1)) / 2)

    if pooled_sd == 0:
        return 0.0
    return (mean_A - mean_B) / pooled_sd


def comprehensive_stability_metrics(
    pred_matrix: np.ndarray,
    is_classification: bool = False,
    y_true: np.ndarray = None,
    compute_ci: bool = False,
    n_bootstrap: int = 2000,
    seed: int = 42,
):
    """
    Compute comprehensive stability metrics.

    Args:
        pred_matrix: (n_fits, n_samples) for regression,
                     (n_fits, n_samples, n_classes) for classification
        is_classification: whether this is a classification task
        y_true: true labels for classification tasks
        compute_ci: whether to compute bootstrap CIs (slower but provides SEs)
        n_bootstrap: number of bootstrap samples if compute_ci=True
        seed: random seed for bootstrap

    Returns:
        dict with stability metrics (and SEs/CIs if compute_ci=True)
    """
    metrics = {}

    if not is_classification:
        per_point_var = pred_matrix.var(axis=0)
        per_point_std = np.sqrt(per_point_var)

        metrics["stability_rmse"] = math.sqrt(per_point_var.mean())
        metrics["std_p50"] = np.percentile(per_point_std, 50)
        metrics["std_p90"] = np.percentile(per_point_std, 90)
        metrics["std_p95"] = np.percentile(per_point_std, 95)
        metrics["std_max"] = per_point_std.max()

        if compute_ci:
            est = stability_rmse_with_ci(pred_matrix, n_bootstrap=n_bootstrap, seed=seed)
            metrics["stability_rmse_se"] = est.se
            metrics["stability_rmse_ci_lower"] = est.ci_lower
            metrics["stability_rmse_ci_upper"] = est.ci_upper

    else:
        if pred_matrix.ndim == 3:
            logits = pred_matrix
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            pred_classes = logits.argmax(axis=-1)
        else:
            probs = pred_matrix
            pred_classes = (pred_matrix > 0.5).astype(int)

        if pred_matrix.ndim == 3:
            logit_var = logits.var(axis=0).mean(axis=-1)
            metrics["logit_stability_rmse"] = math.sqrt(logit_var.mean())
            metrics["logit_std_p50"] = np.percentile(np.sqrt(logit_var), 50)
            metrics["logit_std_p90"] = np.percentile(np.sqrt(logit_var), 90)
            metrics["logit_std_p95"] = np.percentile(np.sqrt(logit_var), 95)
            metrics["logit_std_max"] = np.sqrt(logit_var).max()

            if compute_ci:
                est = logit_stability_with_ci(logits, n_bootstrap=n_bootstrap, seed=seed)
                metrics["logit_stability_rmse_se"] = est.se
                metrics["logit_stability_rmse_ci_lower"] = est.ci_lower
                metrics["logit_stability_rmse_ci_upper"] = est.ci_upper

            prob_class1 = probs[:, :, 1]
            prob_var = prob_class1.var(axis=0)
            metrics["prob_stability_rmse"] = math.sqrt(prob_var.mean())
            metrics["prob_std_p50"] = np.percentile(np.sqrt(prob_var), 50)
            metrics["prob_std_p90"] = np.percentile(np.sqrt(prob_var), 90)
            metrics["prob_std_p95"] = np.percentile(np.sqrt(prob_var), 95)
            metrics["prob_std_max"] = np.sqrt(prob_var).max()

        n_fits, n_samples = pred_classes.shape
        mode_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 0, pred_classes
        )
        flip_rates = (pred_classes != mode_pred).mean(axis=0)

        metrics["flip_rate_mean"] = flip_rates.mean()
        metrics["flip_rate_p50"] = np.percentile(flip_rates, 50)
        metrics["flip_rate_p90"] = np.percentile(flip_rates, 90)
        metrics["flip_rate_max"] = flip_rates.max()
        metrics["any_flip_rate"] = (flip_rates > 0).mean()

        if pred_matrix.ndim == 3 and y_true is not None:
            prob_class1 = probs[:, :, 1]
            mean_prob = prob_class1.mean(axis=0)
            margin = np.abs(mean_prob - 0.5)

            low_conf = margin < 0.1
            mid_conf = (margin >= 0.1) & (margin < 0.3)
            high_conf = margin >= 0.3

            if low_conf.sum() > 0:
                metrics["flip_rate_low_conf"] = flip_rates[low_conf].mean()
            if mid_conf.sum() > 0:
                metrics["flip_rate_mid_conf"] = flip_rates[mid_conf].mean()
            if high_conf.sum() > 0:
                metrics["flip_rate_high_conf"] = flip_rates[high_conf].mean()

    return metrics


def classification_stability_analysis(logits_matrix: np.ndarray, y_true: np.ndarray):
    """
    Detailed classification stability analysis.

    Args:
        logits_matrix: (n_fits, n_samples, n_classes) array of logit predictions
        y_true: true labels

    Returns:
        dict with detailed stability analysis
    """
    n_fits, n_samples, n_classes = logits_matrix.shape
    probs = torch.softmax(torch.tensor(logits_matrix), dim=-1).numpy()
    pred_classes = logits_matrix.argmax(axis=-1)

    analysis = {}

    logit_var = logits_matrix.var(axis=0)
    analysis["logit_var_per_class"] = logit_var.mean(axis=0).tolist()
    analysis["logit_total_var"] = logit_var.sum(axis=-1).mean()

    prob_var = probs.var(axis=0)
    analysis["prob_var_per_class"] = prob_var.mean(axis=0).tolist()
    analysis["prob_total_var"] = prob_var.sum(axis=-1).mean()

    mode_pred = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=n_classes).argmax(), 0, pred_classes
    )
    disagreement_count = (pred_classes != mode_pred).sum(axis=0)

    analysis["label_flip_rate"] = (disagreement_count > 0).mean()
    analysis["label_disagreement_mean"] = disagreement_count.mean() / n_fits

    correct_mask = mode_pred == y_true
    if correct_mask.sum() > 0:
        analysis["flip_rate_correct"] = (disagreement_count[correct_mask] > 0).mean()
    if (~correct_mask).sum() > 0:
        analysis["flip_rate_incorrect"] = (
            disagreement_count[~correct_mask] > 0
        ).mean()

    return analysis
