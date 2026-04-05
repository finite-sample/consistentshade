#!/usr/bin/env python3
"""Comprehensive simulation experiments for BCR.

Includes:
- P0: K/Lambda grid search
- P0: Extended challenging scenarios
- P1: Sample size scaling
"""

import os
from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm

from bcr import (
    EXPERIMENT_CONFIG,
    comprehensive_stability_metrics,
    set_seed,
    stability_rmse,
    train_baseline_regression,
    train_bcr_classification,
    train_bcr_regression,
)

from .datasets import (
    create_challenging_datasets,
    prepare_adult_income,
    prepare_california_housing,
    prepare_german_credit,
    prepare_synthetic_regression,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABS_DIR = os.path.join(BASE_DIR, "tabs")


def run_k_lambda_grid(datasets, K_values, lam_values, R=30):
    """
    Full hyperparameter grid for BCR.

    Args:
        datasets: dict mapping name -> (train_ds, test_x, test_y, d_in, is_classification)
        K_values: list of K values to test
        lam_values: list of lambda values to test
        R: number of replicates

    Returns:
        DataFrame with results for all (K, lambda, dataset) combinations
    """
    results = []

    configs = [
        (ds_name, K, lam)
        for ds_name in datasets
        for K, lam in product(K_values, lam_values)
    ]

    for ds_name, K, lam in tqdm(configs, desc="K/Lambda Grid"):
        train_ds, test_x, test_y, d_in, is_cls = datasets[ds_name]

        preds_list = []
        metrics_list = []

        for r in range(R):
            seed = EXPERIMENT_CONFIG.base_seed + r

            if is_cls:
                pred, metric = train_bcr_classification(
                    seed, train_ds, test_x, test_y, d_in, K=K, lam=lam
                )
            else:
                pred, metric = train_bcr_regression(
                    seed, train_ds, test_x, test_y, d_in, K=K, lam=lam
                )

            preds_list.append(pred)
            metrics_list.append(metric)

        preds = np.stack(preds_list)
        comp_metrics = comprehensive_stability_metrics(
            preds,
            is_classification=is_cls,
            y_true=test_y.numpy() if is_cls else None,
            compute_ci=True,
        )

        if is_cls:
            stability = comp_metrics.get("logit_stability_rmse", np.nan)
            stability_se = comp_metrics.get("logit_stability_rmse_se", np.nan)
        else:
            stability = comp_metrics["stability_rmse"]
            stability_se = comp_metrics.get("stability_rmse_se", np.nan)

        results.append({
            "dataset": ds_name,
            "K": K,
            "lambda": lam,
            "stability": stability,
            "stability_se": stability_se,
            "avg_metric": np.mean(metrics_list),
            "std_metric": np.std(metrics_list),
            "is_classification": is_cls,
        })

    return pd.DataFrame(results)


def create_extended_regression_scenarios():
    """Create extended challenging regression scenarios."""
    scenarios = {}

    set_seed(100)
    X, y = make_regression(n_samples=50, n_features=10, n_informative=8, noise=10.0, random_state=100)
    X = StandardScaler().fit_transform(X).astype("float32")
    y = y.astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    scenarios["very_low_n"] = {
        "train_ds": TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        "test_x": torch.tensor(X_te),
        "test_y": torch.tensor(y_te),
        "d_in": X.shape[1],
        "description": "Very low n (50), p=10",
    }

    set_seed(101)
    X, y = make_regression(n_samples=200, n_features=20, n_informative=15, noise=15.0, random_state=101)
    X = StandardScaler().fit_transform(X).astype("float32")
    y = y.astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    scenarios["low_n"] = {
        "train_ds": TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        "test_x": torch.tensor(X_te),
        "test_y": torch.tensor(y_te),
        "d_in": X.shape[1],
        "description": "Low n (200), p=20",
    }

    set_seed(102)
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=20.0, random_state=102)
    X = StandardScaler().fit_transform(X).astype("float32")
    y = y.astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    scenarios["moderate_n"] = {
        "train_ds": TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        "test_x": torch.tensor(X_te),
        "test_y": torch.tensor(y_te),
        "d_in": X.shape[1],
        "description": "Moderate n (1000), p=20",
    }

    set_seed(103)
    X, y = make_regression(n_samples=200, n_features=50, n_informative=10, noise=15.0, random_state=103)
    X = StandardScaler().fit_transform(X).astype("float32")
    y = y.astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    scenarios["low_n_high_p"] = {
        "train_ds": TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        "test_x": torch.tensor(X_te),
        "test_y": torch.tensor(y_te),
        "d_in": X.shape[1],
        "description": "Low n (200), high p (50)",
    }

    set_seed(104)
    X, y = make_regression(n_samples=500, n_features=100, n_informative=20, noise=20.0, random_state=104)
    X = StandardScaler().fit_transform(X).astype("float32")
    y = y.astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    scenarios["very_high_p"] = {
        "train_ds": TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        "test_x": torch.tensor(X_te),
        "test_y": torch.tensor(y_te),
        "d_in": X.shape[1],
        "description": "Very high p (100), n=500",
    }

    for rho, label in [(0.5, "mild_correlation"), (0.9, "strong_correlation")]:
        set_seed(105 + int(rho * 10))
        n, p = 1000, 20
        cov = np.eye(p)
        for i in range(p):
            for j in range(p):
                cov[i, j] = rho ** abs(i - j)
        X = np.random.multivariate_normal(np.zeros(p), cov, size=n).astype("float32")
        beta = np.random.randn(p).astype("float32")
        y = (X @ beta + np.random.randn(n) * 5).astype("float32")
        X = StandardScaler().fit_transform(X).astype("float32")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
        scenarios[label] = {
            "train_ds": TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            "test_x": torch.tensor(X_te),
            "test_y": torch.tensor(y_te),
            "d_in": X.shape[1],
            "description": f"Correlated features (rho={rho})",
        }

    for noise_pct in [10, 20, 40]:
        set_seed(110 + noise_pct)
        X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=10.0, random_state=110 + noise_pct)
        X = StandardScaler().fit_transform(X).astype("float32")
        y = y.astype("float32")
        n_corrupt = int(noise_pct / 100 * len(y))
        corrupt_idx = np.random.choice(len(y), n_corrupt, replace=False)
        y[corrupt_idx] = np.random.randn(n_corrupt) * y.std() * 2
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
        scenarios[f"noise_{noise_pct}pct"] = {
            "train_ds": TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            "test_x": torch.tensor(X_te),
            "test_y": torch.tensor(y_te),
            "d_in": X.shape[1],
            "description": f"{noise_pct}% label noise",
        }

    return scenarios


def run_challenging_scenarios(R=30):
    """Run extended ablation on difficult settings."""
    scenarios = create_extended_regression_scenarios()

    results = []

    for name, data in tqdm(scenarios.items(), desc="Challenging Scenarios"):
        baseline_preds = []
        baseline_metrics = []
        bcr_preds = []
        bcr_metrics = []

        for r in range(R):
            seed = EXPERIMENT_CONFIG.base_seed + r

            bp, bm = train_baseline_regression(
                seed, data["train_ds"], data["test_x"], data["test_y"], data["d_in"]
            )
            kp, km = train_bcr_regression(
                seed, data["train_ds"], data["test_x"], data["test_y"], data["d_in"]
            )

            baseline_preds.append(bp)
            baseline_metrics.append(bm)
            bcr_preds.append(kp)
            bcr_metrics.append(km)

        baseline_preds = np.stack(baseline_preds)
        bcr_preds = np.stack(bcr_preds)

        baseline_stab = stability_rmse(baseline_preds)
        bcr_stab = stability_rmse(bcr_preds)

        improvement_pct = (baseline_stab - bcr_stab) / baseline_stab * 100

        results.append({
            "scenario": name,
            "description": data["description"],
            "baseline_rmse": np.mean(baseline_metrics),
            "bcr_rmse": np.mean(bcr_metrics),
            "baseline_stability": baseline_stab,
            "bcr_stability": bcr_stab,
            "stability_improvement_pct": improvement_pct,
            "rmse_change_pct": (np.mean(bcr_metrics) - np.mean(baseline_metrics)) / np.mean(baseline_metrics) * 100,
        })

    return pd.DataFrame(results)


def run_sample_size_scaling(fractions, R=30):
    """
    Test how BCR benefit scales with sample size.

    For each fraction, subsample the training data and compare BCR vs Baseline.
    """
    print("Preparing full California Housing dataset...")
    full_ds, test_x, test_y, d_in = prepare_california_housing()

    X_full = full_ds.tensors[0].numpy()
    y_full = full_ds.tensors[1].numpy()
    n_full = len(X_full)

    results = []

    for frac in tqdm(fractions, desc="Sample Size Scaling"):
        n_subset = int(n_full * frac)

        set_seed(42)
        idx = np.random.choice(n_full, n_subset, replace=False)
        subset_ds = TensorDataset(torch.tensor(X_full[idx]), torch.tensor(y_full[idx]))

        baseline_preds = []
        baseline_metrics = []
        bcr_preds = []
        bcr_metrics = []

        for r in range(R):
            seed = EXPERIMENT_CONFIG.base_seed + r

            bp, bm = train_baseline_regression(seed, subset_ds, test_x, test_y, d_in)
            kp, km = train_bcr_regression(seed, subset_ds, test_x, test_y, d_in)

            baseline_preds.append(bp)
            baseline_metrics.append(bm)
            bcr_preds.append(kp)
            bcr_metrics.append(km)

        baseline_preds = np.stack(baseline_preds)
        bcr_preds = np.stack(bcr_preds)

        baseline_stab = stability_rmse(baseline_preds)
        bcr_stab = stability_rmse(bcr_preds)

        results.append({
            "fraction": frac,
            "n_train": n_subset,
            "baseline_rmse": np.mean(baseline_metrics),
            "bcr_rmse": np.mean(bcr_metrics),
            "baseline_stability": baseline_stab,
            "bcr_stability": bcr_stab,
            "stability_improvement_pct": (baseline_stab - bcr_stab) / baseline_stab * 100,
        })

    return pd.DataFrame(results)


def main():
    comprehensive_dir = os.path.join(TABS_DIR, "comprehensive")
    os.makedirs(comprehensive_dir, exist_ok=True)

    R = EXPERIMENT_CONFIG.n_replicates

    print("=" * 80)
    print("P0: K/Lambda Grid Search")
    print("=" * 80)

    K_values = [2, 3, 5, 7, 10]
    lam_values = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    print("\nPreparing datasets...")
    syn_ds, syn_test_x, syn_test_y, syn_d_in = prepare_synthetic_regression()
    cal_ds, cal_test_x, cal_test_y, cal_d_in = prepare_california_housing()
    adult_ds, adult_test_x, adult_test_y, adult_d_in = prepare_adult_income()
    german_ds, german_test_x, german_test_y, german_d_in = prepare_german_credit()

    datasets = {
        "synthetic": (syn_ds, syn_test_x, syn_test_y, syn_d_in, False),
        "california": (cal_ds, cal_test_x, cal_test_y, cal_d_in, False),
        "adult": (adult_ds, adult_test_x, adult_test_y, adult_d_in, True),
        "german": (german_ds, german_test_x, german_test_y, german_d_in, True),
    }

    print(f"\nGrid: K={K_values}, lambda={lam_values}")
    print(f"Total: {len(K_values) * len(lam_values) * len(datasets)} configurations")

    k_lambda_results = run_k_lambda_grid(datasets, K_values, lam_values, R=R)
    k_lambda_results.to_csv(os.path.join(comprehensive_dir, "k_lambda_grid.csv"), index=False)

    print("\n" + "=" * 80)
    print("P0: Challenging Scenarios")
    print("=" * 80)

    scenario_results = run_challenging_scenarios(R=R)
    scenario_results.to_csv(os.path.join(comprehensive_dir, "challenging_scenarios.csv"), index=False)

    print("\nScenario Results:")
    print(scenario_results[["scenario", "stability_improvement_pct", "rmse_change_pct"]].round(2).to_string(index=False))

    print("\n" + "=" * 80)
    print("P1: Sample Size Scaling")
    print("=" * 80)

    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    scaling_results = run_sample_size_scaling(fractions, R=R)
    scaling_results.to_csv(os.path.join(comprehensive_dir, "sample_size_scaling.csv"), index=False)

    print("\nSample Size Scaling Results:")
    print(scaling_results.round(4).to_string(index=False))

    print(f"\n\nAll results saved to {comprehensive_dir}/")


if __name__ == "__main__":
    main()
