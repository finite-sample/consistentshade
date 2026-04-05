#!/usr/bin/env python3
"""Statistical analysis of experiment results.

Loads raw predictions and computes:
1. Bootstrap confidence intervals for stability metrics
2. Pairwise significance tests between methods
3. Cohen's d effect sizes
4. Publication-ready summary tables
"""

import os

import numpy as np
import pandas as pd

from bcr import (
    MethodComparison,
    cohens_d,
    compare_methods_bootstrap,
    logit_stability_rmse,
    logit_stability_with_ci,
    stability_rmse_with_ci,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABS_DIR = os.path.join(BASE_DIR, "tabs")


def load_raw_predictions(dataset_name):
    """Load raw predictions from npz file."""
    raw_dir = os.path.join(TABS_DIR, "raw_predictions")
    filepath = os.path.join(raw_dir, f"{dataset_name}_raw.npz")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw predictions not found: {filepath}")
    return dict(np.load(filepath))


def get_method_names():
    """Return list of method names."""
    return ["baseline", "bcr", "weight_decay", "sam", "r_drop", "bagging", "ifr", "ifr_kfac"]


def analyze_dataset(dataset_name, is_classification=False, n_bootstrap=2000):
    """
    Comprehensive statistical analysis of a single dataset.

    Returns:
        summary_df: DataFrame with point estimates, SEs, and CIs
        pairwise_df: DataFrame with pairwise significance tests vs baseline
    """
    data = load_raw_predictions(dataset_name)
    methods = get_method_names()

    summary_rows = []
    pairwise_rows = []

    baseline_preds = data["baseline_preds"]

    for method in methods:
        preds_key = f"{method}_preds"
        metrics_key = f"{method}_metrics"

        if preds_key not in data:
            continue

        preds = data[preds_key]
        metrics = data[metrics_key]

        if is_classification:
            est = logit_stability_with_ci(preds, n_bootstrap=n_bootstrap)
            stability_val = est.point
        else:
            est = stability_rmse_with_ci(preds, n_bootstrap=n_bootstrap)
            stability_val = est.point

        summary_rows.append({
            "Method": method.replace("_", " ").title(),
            "Stability": stability_val,
            "SE": est.se,
            "CI_Lower": est.ci_lower,
            "CI_Upper": est.ci_upper,
            "Avg_Metric": float(np.mean(metrics)),
            "Std_Metric": float(np.std(metrics)),
        })

        if method != "baseline":
            if is_classification:
                comp = compare_logits_bootstrap(baseline_preds, preds, n_bootstrap=n_bootstrap)
            else:
                comp = compare_methods_bootstrap(baseline_preds, preds, n_bootstrap=n_bootstrap)

            effect_size = cohens_d(baseline_preds, preds)

            pairwise_rows.append({
                "Method": method.replace("_", " ").title(),
                "Diff_vs_Baseline": comp.diff,
                "SE": comp.se,
                "CI_Lower": comp.ci_lower,
                "CI_Upper": comp.ci_upper,
                "p_value": comp.p_value,
                "Significant_05": comp.significant_at_05,
                "Significant_01": comp.significant_at_01,
                "Cohens_d": effect_size,
            })

    summary_df = pd.DataFrame(summary_rows)
    pairwise_df = pd.DataFrame(pairwise_rows) if pairwise_rows else None

    return summary_df, pairwise_df


def compare_logits_bootstrap(logits_A, logits_B, n_bootstrap=2000, ci=0.95, seed=42):
    """Bootstrap test for difference in logit stability."""
    rng = np.random.RandomState(seed)
    n_fits = logits_A.shape[0]

    boot_diffs = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n_fits, n_fits, replace=True)
        stab_A = logit_stability_rmse(logits_A[idx])
        stab_B = logit_stability_rmse(logits_B[idx])
        boot_diffs[b] = stab_A - stab_B

    point_diff = logit_stability_rmse(logits_A) - logit_stability_rmse(logits_B)
    se = np.std(boot_diffs, ddof=1)

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(boot_diffs, alpha * 100)
    ci_upper = np.percentile(boot_diffs, (1 - alpha) * 100)

    p_value = 2 * min(np.mean(boot_diffs >= 0), np.mean(boot_diffs <= 0))
    p_value = min(p_value, 1.0)

    return MethodComparison(
        diff=float(point_diff),
        se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(p_value),
    )


def format_significance(p_value):
    """Format p-value with significance stars."""
    if p_value < 0.001:
        return f"{p_value:.2e}***"
    elif p_value < 0.01:
        return f"{p_value:.3f}**"
    elif p_value < 0.05:
        return f"{p_value:.3f}*"
    else:
        return f"{p_value:.3f}"


def generate_latex_table(summary_df, pairwise_df, caption, label):
    """Generate LaTeX table with significance markers."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Stability & SE & 95\% CI & vs Baseline $p$ & Cohen's $d$ \\",
        r"\midrule",
    ]

    for _, row in summary_df.iterrows():
        method = row["Method"]
        stability = f"{row['Stability']:.4f}"
        se = f"{row['SE']:.4f}"
        ci = f"[{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}]"

        if pairwise_df is not None and method != "Baseline":
            pw = pairwise_df[pairwise_df["Method"] == method]
            if len(pw) > 0:
                pw_row = pw.iloc[0]
                p_str = format_significance(pw_row["p_value"])
                d_str = f"{pw_row['Cohens_d']:.2f}"
            else:
                p_str = "-"
                d_str = "-"
        else:
            p_str = "-"
            d_str = "-"

        lines.append(f"{method} & {stability} & {se} & {ci} & {p_str} & {d_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    print("Statistical Analysis of BCR Experiments")
    print("=" * 60)

    output_dir = os.path.join(TABS_DIR, "statistical_analysis")
    os.makedirs(output_dir, exist_ok=True)

    datasets = [
        ("synthetic", False, "Synthetic Regression"),
        ("california", False, "California Housing"),
        ("adult", True, "Adult Income"),
        ("german", True, "German Credit"),
    ]

    for dataset_name, is_classification, display_name in datasets:
        print(f"\n{display_name}")
        print("-" * 40)

        try:
            summary_df, pairwise_df = analyze_dataset(
                dataset_name, is_classification=is_classification
            )

            print("\nSummary Statistics:")
            print(summary_df.to_string(index=False))

            if pairwise_df is not None:
                print("\nPairwise Comparisons vs Baseline:")
                print(pairwise_df.to_string(index=False))

            summary_df.to_csv(
                os.path.join(output_dir, f"{dataset_name}_summary.csv"), index=False
            )
            if pairwise_df is not None:
                pairwise_df.to_csv(
                    os.path.join(output_dir, f"{dataset_name}_pairwise.csv"), index=False
                )

            latex = generate_latex_table(
                summary_df,
                pairwise_df,
                caption=f"Statistical analysis of stability metrics ({display_name}). "
                        "SE = bootstrap standard error. $p$-values from bootstrap test vs baseline. "
                        "***$p<0.001$, **$p<0.01$, *$p<0.05$.",
                label=f"tab:{dataset_name}_stats",
            )

            with open(os.path.join(output_dir, f"{dataset_name}_table.tex"), "w") as f:
                f.write(latex)

        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
