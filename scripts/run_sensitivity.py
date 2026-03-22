#!/usr/bin/env python3
"""Hyperparameter sensitivity analysis for BCR.

Tests sensitivity to:
- K (number of shadow models): [2, 3, 5, 7]
- Batch size: [32, 64, 128, 256]
- Learning rate: [1e-4, 1e-3, 1e-2]
- Hidden dimension: [32, 64, 128]

Runs on 2 datasets: Synthetic (regression), Adult Income (classification)
Uses R=30 replicates for each configuration.
"""

import os
import sys
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import EXPERIMENT_CONFIG, TABS_DIR
from scripts.datasets import prepare_adult_income, prepare_synthetic_regression
from scripts.metrics import comprehensive_stability_metrics
from scripts.trainers import train_bcr_classification, train_bcr_regression


def run_sensitivity_experiment(
    train_ds,
    test_x,
    test_y,
    d_in,
    is_classification,
    K_values,
    bs_values,
    lr_values,
    hid_values,
    R=30,
    lam=0.05,
):
    """Run sensitivity analysis over hyperparameter grid."""
    results = []

    total_configs = len(K_values) * len(bs_values) * len(lr_values) * len(hid_values)
    config_idx = 0

    for K, bs, lr, hid in product(K_values, bs_values, lr_values, hid_values):
        config_idx += 1
        print(f"  Config {config_idx}/{total_configs}: K={K}, bs={bs}, lr={lr}, hid={hid}")

        preds_list = []
        metrics_list = []

        for r in range(R):
            seed = EXPERIMENT_CONFIG["base_seed"] + r

            if is_classification:
                pred, metric = train_bcr_classification(
                    seed, train_ds, test_x, test_y, d_in,
                    K=K, lam=lam, bs=bs, lr=lr, hid=hid,
                )
            else:
                pred, metric = train_bcr_regression(
                    seed, train_ds, test_x, test_y, d_in,
                    K=K, lam=lam, bs=bs, lr=lr, hid=hid,
                )

            preds_list.append(pred)
            metrics_list.append(metric)

        preds = np.stack(preds_list)
        comp_metrics = comprehensive_stability_metrics(
            preds,
            is_classification=is_classification,
            y_true=test_y.numpy() if is_classification else None,
            compute_ci=True,
        )

        if is_classification:
            stability = comp_metrics.get("logit_stability_rmse", np.nan)
            stability_se = comp_metrics.get("logit_stability_rmse_se", np.nan)
        else:
            stability = comp_metrics["stability_rmse"]
            stability_se = comp_metrics.get("stability_rmse_se", np.nan)

        results.append({
            "K": K,
            "batch_size": bs,
            "learning_rate": lr,
            "hidden_dim": hid,
            "stability": stability,
            "stability_se": stability_se,
            "avg_metric": np.mean(metrics_list),
            "std_metric": np.std(metrics_list),
        })

    return pd.DataFrame(results)


def main():
    R = EXPERIMENT_CONFIG["n_replicates"]
    lam = 0.05

    K_values = [2, 3, 5, 7]
    bs_values = [32, 64, 128, 256]
    lr_values = [1e-4, 1e-3, 1e-2]
    hid_values = [32, 64, 128]

    print("Preparing datasets...")
    syn_ds, syn_test_x, syn_test_y, syn_d_in = prepare_synthetic_regression()
    adult_ds, adult_test_x, adult_test_y, adult_d_in = prepare_adult_income()

    print(f"\nRunning sensitivity analysis ({R} replicates per config)...")
    print(f"Grid: K={K_values}, bs={bs_values}, lr={lr_values}, hid={hid_values}")
    print(f"Total configurations: {len(K_values) * len(bs_values) * len(lr_values) * len(hid_values)}")

    print("\n1. Synthetic Regression")
    syn_results = run_sensitivity_experiment(
        syn_ds, syn_test_x, syn_test_y, syn_d_in,
        is_classification=False,
        K_values=K_values,
        bs_values=bs_values,
        lr_values=lr_values,
        hid_values=hid_values,
        R=R,
        lam=lam,
    )

    print("\n2. Adult Income Classification")
    adult_results = run_sensitivity_experiment(
        adult_ds, adult_test_x, adult_test_y, adult_d_in,
        is_classification=True,
        K_values=K_values,
        bs_values=bs_values,
        lr_values=lr_values,
        hid_values=hid_values,
        R=R,
        lam=lam,
    )

    # Save results
    sensitivity_dir = os.path.join(TABS_DIR, "sensitivity")
    os.makedirs(sensitivity_dir, exist_ok=True)

    syn_results.to_csv(os.path.join(sensitivity_dir, "synthetic_sensitivity.csv"), index=False)
    adult_results.to_csv(os.path.join(sensitivity_dir, "adult_sensitivity.csv"), index=False)

    print(f"\nSensitivity results saved to {sensitivity_dir}/")

    # Print summary tables
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 80)

    print("\nSynthetic Regression - Effect of K (averaging over other params):")
    print(syn_results.groupby("K")[["stability", "avg_metric"]].mean().round(4))

    print("\nSynthetic Regression - Effect of batch_size:")
    print(syn_results.groupby("batch_size")[["stability", "avg_metric"]].mean().round(4))

    print("\nAdult Classification - Effect of K:")
    print(adult_results.groupby("K")[["stability", "avg_metric"]].mean().round(4))

    print("\nAdult Classification - Effect of batch_size:")
    print(adult_results.groupby("batch_size")[["stability", "avg_metric"]].mean().round(4))


if __name__ == "__main__":
    main()
