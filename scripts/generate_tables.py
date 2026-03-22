#!/usr/bin/env python3
"""Generate LaTeX tables from experiment results."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import TABS_DIR


def format_with_se(val, se, is_large=False):
    """Format value with SE in parentheses."""
    if pd.isna(se):
        if is_large:
            return f"{val:.2f}"
        return f"{val:.3f}"
    if is_large:
        return f"{val:.2f} ({se:.2f})"
    return f"{val:.3f} ({se:.3f})"


def generate_main_results_table():
    """Generate main results LaTeX table with SEs."""
    syn_df = pd.read_csv(os.path.join(TABS_DIR, "synthetic_results.csv"))
    cal_df = pd.read_csv(os.path.join(TABS_DIR, "california_results.csv"))
    adult_df = pd.read_csv(os.path.join(TABS_DIR, "adult_results.csv"))
    credit_df = pd.read_csv(os.path.join(TABS_DIR, "german_results.csv"))

    datasets = [
        ("Synthetic", syn_df, False),
        ("California", cal_df, False),
        ("Adult", adult_df, True),
        ("German", credit_df, True),
    ]

    methods_order = ["Baseline", "Weight Decay", "SAM", "R-Drop", "Bagging", "IFR", "BCR"]

    with open(os.path.join(TABS_DIR, "main_results.tex"), "w") as f:
        f.write("\\begin{tabular}{llccccccc}\n")
        f.write("\\toprule\n")
        f.write(
            "Dataset & Metric & Baseline & Weight Decay & SAM & R-Drop & Bagging & IFR & \\textbf{BCR} \\\\\n"
        )
        f.write("\\midrule\n")

        for ds_name, summary, is_class in datasets:
            perf_metric = "Accuracy $\\uparrow$" if is_class else "RMSE $\\downarrow$"
            perf_col = "Avg Accuracy" if is_class else "Avg RMSE"
            stab_col = "Logit Stability" if is_class else "Stability RMSE"
            se_col = "Logit Stability SE" if is_class else "Stability SE"

            perf_vals = []
            stab_vals = []
            stab_ses = []
            for method in methods_order:
                row = summary[summary["Method"] == method]
                if len(row) > 0:
                    perf_vals.append(row[perf_col].iloc[0])
                    stab_vals.append(row[stab_col].iloc[0])
                    if se_col in row.columns:
                        stab_ses.append(row[se_col].iloc[0])
                    else:
                        stab_ses.append(float("nan"))
                else:
                    perf_vals.append(float("nan"))
                    stab_vals.append(float("nan"))
                    stab_ses.append(float("nan"))

            if is_class:
                perf_str = " & ".join([f"{v:.3f}" for v in perf_vals])
            else:
                perf_str = " & ".join(
                    [f"{v:.2f}" if v > 1 else f"{v:.3f}" for v in perf_vals]
                )

            # Include SE in stability values
            is_large = not is_class and any(v > 1 for v in stab_vals if not pd.isna(v))
            stab_strs = []
            for v, se in zip(stab_vals, stab_ses):
                if pd.isna(v):
                    stab_strs.append("-")
                elif pd.isna(se):
                    stab_strs.append(f"{v:.2f}" if is_large else f"{v:.3f}")
                else:
                    stab_strs.append(format_with_se(v, se, is_large=is_large))
            stab_str = " & ".join(stab_strs)

            f.write(f"{ds_name} & {perf_metric} & {perf_str} \\\\\n")
            f.write(f" & Stability (SE) & {stab_str} \\\\\n")

            if ds_name != "German":
                f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"Generated: {os.path.join(TABS_DIR, 'main_results.tex')}")


def generate_classification_stability_table():
    """Generate detailed classification stability table with SEs."""
    adult_df = pd.read_csv(os.path.join(TABS_DIR, "adult_results.csv"))

    methods_order = ["Baseline", "BCR", "Weight Decay", "SAM", "R-Drop", "Bagging"]

    with open(os.path.join(TABS_DIR, "classification_stability.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Logit Stability (SE) & Prob Stability & Flip Rate \\\\\n")
        f.write("\\midrule\n")

        for method in methods_order:
            row = adult_df[adult_df["Method"] == method]
            if len(row) == 0:
                continue

            logit_stab = row["Logit Stability"].iloc[0]
            logit_se = row.get("Logit Stability SE", pd.Series([float("nan")])).iloc[0]
            prob_stab = row["Prob Stability"].iloc[0]
            flip_rate = row["Flip Rate"].iloc[0]

            if pd.isna(logit_se):
                logit_str = f"{logit_stab:.3f}"
            else:
                logit_str = f"{logit_stab:.3f} ({logit_se:.3f})"

            f.write(f"{method} & {logit_str} & {prob_stab:.4f} & {flip_rate:.4f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"Generated: {os.path.join(TABS_DIR, 'classification_stability.tex')}")


def generate_ablation_table():
    """Generate ablation study LaTeX table."""
    ablation_df = pd.read_csv(os.path.join(TABS_DIR, "ablation_results.csv"))

    with open(os.path.join(TABS_DIR, "ablation_results.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write(
            "Dataset & \\multicolumn{2}{c}{RMSE} & \\multicolumn{2}{c}{Stability} & \\multicolumn{2}{c}{$\\Delta$\\%} \\\\\n"
        )
        f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n")
        f.write(" & Baseline & BCR & Baseline & BCR & BCR & IFR \\\\\n")
        f.write("\\midrule\n")

        for _, row in ablation_df.iterrows():
            f.write(
                f"{row['Dataset']} & {row['Baseline RMSE']:.2f} & {row['BCR RMSE']:.2f} & "
                f"{row['Baseline Stability']:.2f} & {row['BCR Stability']:.2f} & "
                f"{row['BCR Stability Δ%']:.1f} & {row['IFR Stability Δ%']:.1f} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"Generated: {os.path.join(TABS_DIR, 'ablation_results.tex')}")


def main():
    if not os.path.exists(TABS_DIR):
        print(f"Error: {TABS_DIR} does not exist. Run experiments first.")
        sys.exit(1)

    required_files = [
        "synthetic_results.csv",
        "california_results.csv",
        "adult_results.csv",
        "german_results.csv",
    ]

    missing = [f for f in required_files if not os.path.exists(os.path.join(TABS_DIR, f))]
    if missing:
        print(f"Error: Missing result files: {missing}")
        print("Run run_main_experiments.py first.")
        sys.exit(1)

    generate_main_results_table()
    generate_classification_stability_table()

    if os.path.exists(os.path.join(TABS_DIR, "ablation_results.csv")):
        generate_ablation_table()
    else:
        print("Skipping ablation table (run_ablation.py not run yet)")


if __name__ == "__main__":
    main()
