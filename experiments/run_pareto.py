#!/usr/bin/env python3
"""Generate Pareto frontier plots for BCR lambda sweep."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bcr import stability_rmse, train_baseline_regression, train_bcr_regression

from .datasets import prepare_california_housing, prepare_synthetic_regression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABS_DIR = os.path.join(BASE_DIR, "tabs")
FIGS_DIR = os.path.join(BASE_DIR, "figs")


def pareto_sweep(train_ds, test_x, test_y, d_in, R=10):
    """Sweep lambda values to generate Pareto frontier data."""
    lambdas = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    baseline_preds = []
    baseline_metrics = []
    for r in range(R):
        pred, metric = train_baseline_regression(12345 + r, train_ds, test_x, test_y, d_in)
        baseline_preds.append(pred)
        baseline_metrics.append(metric)

    baseline_preds = np.stack(baseline_preds)
    baseline_stability = stability_rmse(baseline_preds)
    baseline_perf = np.mean(baseline_metrics)

    sweep_results = [
        {"lambda": "Baseline", "perf": baseline_perf, "stability": baseline_stability}
    ]

    for lam in lambdas:
        print(f"  λ = {lam}")
        preds = []
        metrics = []
        for r in range(R):
            pred, metric = train_bcr_regression(
                12345 + r, train_ds, test_x, test_y, d_in, lam=lam
            )
            preds.append(pred)
            metrics.append(metric)

        preds = np.stack(preds)
        sweep_results.append(
            {"lambda": lam, "perf": np.mean(metrics), "stability": stability_rmse(preds)}
        )

    return pd.DataFrame(sweep_results)


def plot_pareto_frontier(sweep_df, title, save_path=None):
    """Plot Pareto frontier from sweep results."""
    fig, ax = plt.subplots(figsize=(8, 6))

    baseline = sweep_df[sweep_df["lambda"] == "Baseline"].iloc[0]
    ax.scatter(
        baseline["stability"],
        baseline["perf"],
        s=150,
        c="red",
        marker="s",
        label="Baseline",
        zorder=5,
    )

    bcr_df = sweep_df[sweep_df["lambda"] != "Baseline"].copy()
    bcr_df["lambda"] = bcr_df["lambda"].astype(float)

    for _, row in bcr_df.iterrows():
        ax.scatter(row["stability"], row["perf"], s=100, c="blue", alpha=0.7)
        ax.annotate(
            f"λ={row['lambda']}",
            (row["stability"], row["perf"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.plot(bcr_df["stability"], bcr_df["perf"], "b--", alpha=0.5, label="BCR (varying λ)")

    ax.set_xlabel("Prediction Instability (Stability RMSE)", fontsize=12)
    ax.set_ylabel("Test RMSE", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(TABS_DIR, exist_ok=True)

    R = 10

    print("Preparing datasets...")
    syn_ds, syn_test_x, syn_test_y, syn_d_in = prepare_synthetic_regression()
    cal_ds, cal_test_x, cal_test_y, cal_d_in = prepare_california_housing()

    print("\nSweeping lambda on Synthetic...")
    syn_sweep = pareto_sweep(syn_ds, syn_test_x, syn_test_y, syn_d_in, R=R)

    print("\nSweeping lambda on California Housing...")
    cal_sweep = pareto_sweep(cal_ds, cal_test_x, cal_test_y, cal_d_in, R=R)

    plot_pareto_frontier(
        syn_sweep,
        "Pareto Frontier: Synthetic Regression",
        save_path=os.path.join(FIGS_DIR, "pareto_synthetic.png"),
    )

    plot_pareto_frontier(
        cal_sweep,
        "Pareto Frontier: California Housing",
        save_path=os.path.join(FIGS_DIR, "pareto_california.png"),
    )

    syn_sweep.to_csv(os.path.join(TABS_DIR, "pareto_synthetic.csv"), index=False)
    cal_sweep.to_csv(os.path.join(TABS_DIR, "pareto_california.csv"), index=False)

    print("\n" + "=" * 60)
    print("Synthetic Sweep Results:")
    print(syn_sweep.to_string(index=False))
    print("\nCalifornia Sweep Results:")
    print(cal_sweep.to_string(index=False))

    plt.show()


if __name__ == "__main__":
    main()
