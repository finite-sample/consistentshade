#!/usr/bin/env python3
"""Run ablation studies on challenging datasets."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import EXPERIMENT_CONFIG, TABS_DIR
from scripts.datasets import create_challenging_datasets
from scripts.metrics import stability_rmse
from scripts.trainers import (
    train_baseline_regression,
    train_bcr_regression,
    train_ifr_regression,
)


def main():
    os.makedirs(TABS_DIR, exist_ok=True)

    R = EXPERIMENT_CONFIG["n_replicates"]

    print("Creating challenging datasets...")
    challenging = create_challenging_datasets()

    ablation_results = []

    for name, data in challenging.items():
        print(f"\nRunning {name}: {data['description']}")

        baseline_preds = []
        baseline_rmse_list = []
        bcr_preds = []
        bcr_rmse_list = []
        ifr_preds = []
        ifr_rmse_list = []

        for r in range(R):
            seed = EXPERIMENT_CONFIG["base_seed"] + r
            print(f"  Replicate {r + 1}/{R}", end="\r")

            bp, br = train_baseline_regression(
                seed, data["train_ds"], data["test_x"], data["test_y"], data["d_in"]
            )
            kp, kr = train_bcr_regression(
                seed, data["train_ds"], data["test_x"], data["test_y"], data["d_in"]
            )
            ip, ir = train_ifr_regression(
                seed, data["train_ds"], data["test_x"], data["test_y"], data["d_in"]
            )

            baseline_preds.append(bp)
            baseline_rmse_list.append(br)
            bcr_preds.append(kp)
            bcr_rmse_list.append(kr)
            ifr_preds.append(ip)
            ifr_rmse_list.append(ir)

        print()

        baseline_preds = np.stack(baseline_preds)
        bcr_preds = np.stack(bcr_preds)
        ifr_preds = np.stack(ifr_preds)

        baseline_stab = stability_rmse(baseline_preds)
        bcr_stab = stability_rmse(bcr_preds)
        ifr_stab = stability_rmse(ifr_preds)

        ablation_results.append(
            {
                "Dataset": name,
                "Description": data["description"],
                "Baseline RMSE": np.mean(baseline_rmse_list),
                "BCR RMSE": np.mean(bcr_rmse_list),
                "IFR RMSE": np.mean(ifr_rmse_list),
                "Baseline Stability": baseline_stab,
                "BCR Stability": bcr_stab,
                "IFR Stability": ifr_stab,
                "BCR Stability Δ%": (bcr_stab - baseline_stab) / baseline_stab * 100,
                "IFR Stability Δ%": (ifr_stab - baseline_stab) / baseline_stab * 100,
            }
        )

    ablation_df = pd.DataFrame(ablation_results)

    print("\n" + "=" * 80)
    print("Ablation Study Results: Challenging Datasets")
    print("=" * 80)
    print(ablation_df.round(3).to_string(index=False))

    ablation_df.to_csv(os.path.join(TABS_DIR, "ablation_results.csv"), index=False)
    print(f"\nResults saved to {TABS_DIR}/ablation_results.csv")


if __name__ == "__main__":
    main()
