"""Stability metrics for regression and classification."""

import math

import numpy as np
import torch


def stability_rmse(pred_matrix: np.ndarray) -> float:
    """RMSE of predictions across fits: sqrt(mean variance per sample)."""
    return math.sqrt(pred_matrix.var(axis=0).mean())


def comprehensive_stability_metrics(
    pred_matrix: np.ndarray,
    is_classification: bool = False,
    y_true: np.ndarray = None,
):
    """
    Compute comprehensive stability metrics.

    Args:
        pred_matrix: (n_fits, n_samples) for regression,
                     (n_fits, n_samples, n_classes) for classification
        is_classification: whether this is a classification task
        y_true: true labels for classification tasks

    Returns:
        dict with stability metrics
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
