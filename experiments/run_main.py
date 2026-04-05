#!/usr/bin/env python3
"""Run main experiments comparing all methods across datasets."""

import os

import numpy as np
import pandas as pd

from bcr import (
    EXPERIMENT_CONFIG,
    comprehensive_stability_metrics,
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

from .datasets import (
    prepare_adult_income,
    prepare_california_housing,
    prepare_german_credit,
    prepare_synthetic_regression,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABS_DIR = os.path.join(BASE_DIR, "tabs")
FIGS_DIR = os.path.join(BASE_DIR, "figs")


def run_experiment(
    dataset_name, train_ds, test_x, test_y, d_in, is_classification=False, R=30, lam=0.05
):
    """Run all methods on a single dataset."""
    methods = ["Baseline", "BCR", "Weight Decay", "SAM", "R-Drop", "Bagging", "IFR", "IFR-KFAC"]

    if is_classification:
        train_fns = [
            lambda s: train_baseline_classification(s, train_ds, test_x, test_y, d_in),
            lambda s: train_bcr_classification(s, train_ds, test_x, test_y, d_in, lam=lam),
            lambda s: train_wd_classification(s, train_ds, test_x, test_y, d_in),
            lambda s: train_sam_classification(s, train_ds, test_x, test_y, d_in),
            lambda s: train_rdrop_classification(s, train_ds, test_x, test_y, d_in),
            lambda s: train_bagging_classification(s, train_ds, test_x, test_y, d_in),
            lambda s: train_ifr_classification(s, train_ds, test_x, test_y, d_in),
            lambda s: train_ifr_kfac_classification(s, train_ds, test_x, test_y, d_in),
        ]
    else:
        train_fns = [
            lambda s: train_baseline_regression(s, train_ds, test_x, test_y, d_in),
            lambda s: train_bcr_regression(s, train_ds, test_x, test_y, d_in, lam=lam),
            lambda s: train_wd_regression(s, train_ds, test_x, test_y, d_in),
            lambda s: train_sam_regression(s, train_ds, test_x, test_y, d_in),
            lambda s: train_rdrop_regression(s, train_ds, test_x, test_y, d_in),
            lambda s: train_bagging_regression(s, train_ds, test_x, test_y, d_in),
            lambda s: train_ifr_regression(s, train_ds, test_x, test_y, d_in),
            lambda s: train_ifr_kfac_regression(s, train_ds, test_x, test_y, d_in),
        ]

    results = {m: {"preds": [], "metric": []} for m in methods}

    for r in range(R):
        seed = EXPERIMENT_CONFIG.base_seed + r
        print(f"  Replicate {r + 1}/{R}", end="\r")
        for method, train_fn in zip(methods, train_fns):
            pred, metric = train_fn(seed)
            results[method]["preds"].append(pred)
            results[method]["metric"].append(metric)

    print()

    summary_rows = []
    for method in methods:
        preds = np.stack(results[method]["preds"])
        metrics_list = results[method]["metric"]

        if is_classification:
            comp_metrics = comprehensive_stability_metrics(
                preds, is_classification=True, y_true=test_y.numpy(), compute_ci=True
            )
            summary_rows.append(
                {
                    "Method": method,
                    "Avg Accuracy": np.mean(metrics_list),
                    "Std Accuracy": np.std(metrics_list),
                    "Logit Stability": comp_metrics.get("logit_stability_rmse", np.nan),
                    "Logit Stability SE": comp_metrics.get("logit_stability_rmse_se", np.nan),
                    "Logit Stability CI Lower": comp_metrics.get("logit_stability_rmse_ci_lower", np.nan),
                    "Logit Stability CI Upper": comp_metrics.get("logit_stability_rmse_ci_upper", np.nan),
                    "Prob Stability": comp_metrics.get("prob_stability_rmse", np.nan),
                    "Flip Rate": comp_metrics.get("flip_rate_mean", np.nan),
                }
            )
        else:
            comp_metrics = comprehensive_stability_metrics(preds, is_classification=False, compute_ci=True)
            summary_rows.append(
                {
                    "Method": method,
                    "Avg RMSE": np.mean(metrics_list),
                    "Std RMSE": np.std(metrics_list),
                    "Stability RMSE": comp_metrics["stability_rmse"],
                    "Stability SE": comp_metrics.get("stability_rmse_se", np.nan),
                    "Stability CI Lower": comp_metrics.get("stability_rmse_ci_lower", np.nan),
                    "Stability CI Upper": comp_metrics.get("stability_rmse_ci_upper", np.nan),
                    "Std p90": comp_metrics["std_p90"],
                    "Std max": comp_metrics["std_max"],
                }
            )

    return pd.DataFrame(summary_rows), results


def main():
    os.makedirs(TABS_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)

    R = EXPERIMENT_CONFIG.n_replicates

    print("Preparing datasets...")
    syn_ds, syn_test_x, syn_test_y, syn_d_in = prepare_synthetic_regression()
    cal_ds, cal_test_x, cal_test_y, cal_d_in = prepare_california_housing()
    adult_ds, adult_test_x, adult_test_y, adult_d_in = prepare_adult_income()
    credit_ds, credit_test_x, credit_test_y, credit_d_in = prepare_german_credit()

    print(f"\nRunning experiments ({R} replicates each)...\n")

    print("1. Synthetic Regression")
    syn_summary, syn_results = run_experiment(
        "Synthetic", syn_ds, syn_test_x, syn_test_y, syn_d_in, is_classification=False, R=R
    )

    print("2. California Housing")
    cal_summary, cal_results = run_experiment(
        "California", cal_ds, cal_test_x, cal_test_y, cal_d_in, is_classification=False, R=R
    )

    print("3. Adult Income")
    adult_summary, adult_results = run_experiment(
        "Adult", adult_ds, adult_test_x, adult_test_y, adult_d_in, is_classification=True, R=R
    )

    print("4. German Credit")
    credit_summary, credit_results = run_experiment(
        "German", credit_ds, credit_test_x, credit_test_y, credit_d_in, is_classification=True, R=R
    )

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    all_summaries = [
        ("Synthetic Regression", syn_summary),
        ("California Housing", cal_summary),
        ("Adult Income", adult_summary),
        ("German Credit", credit_summary),
    ]

    for name, df in all_summaries:
        print(f"\n{name}:")
        print(df.round(4).to_string(index=False))

    syn_summary.to_csv(os.path.join(TABS_DIR, "synthetic_results.csv"), index=False)
    cal_summary.to_csv(os.path.join(TABS_DIR, "california_results.csv"), index=False)
    adult_summary.to_csv(os.path.join(TABS_DIR, "adult_results.csv"), index=False)
    credit_summary.to_csv(os.path.join(TABS_DIR, "german_results.csv"), index=False)

    RAW_DIR = os.path.join(TABS_DIR, "raw_predictions")
    os.makedirs(RAW_DIR, exist_ok=True)

    def save_raw_preds(results_dict, dataset_name):
        """Save raw predictions as npz file."""
        arrays = {}
        for method, data in results_dict.items():
            key = method.replace(" ", "_").replace("-", "_").lower()
            arrays[f"{key}_preds"] = np.stack(data["preds"])
            arrays[f"{key}_metrics"] = np.array(data["metric"])
        np.savez(os.path.join(RAW_DIR, f"{dataset_name}_raw.npz"), **arrays)

    save_raw_preds(syn_results, "synthetic")
    save_raw_preds(cal_results, "california")
    save_raw_preds(adult_results, "adult")
    save_raw_preds(credit_results, "german")

    print(f"\nResults saved to {TABS_DIR}/")
    print(f"Raw predictions saved to {RAW_DIR}/")


if __name__ == "__main__":
    main()
