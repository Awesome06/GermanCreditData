"""Visualization generation for stress tests, agreement audits, and fairness frontiers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.models import Phase1ComparisonResult
from src.preprocessing import PreprocessedDataset
from src.stress_testing import (
    AdversarialAuditResult,
    FairnessAuditResult,
    OmittedVariableBiasResult,
    SimpsonsParadoxResult,
)


def plot_agreement_confusion_matrix(
    phase1_result: Phase1ComparisonResult,
    output_path: str | Path = "reports/figures/agreement_confusion_matrix.png",
) -> str:
    """Plot publication-grade agreement confusion matrix heatmap."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cm = phase1_result.agreement_confusion_matrix.loc[[0, 1], [0, 1]]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(
        "Model A (OLS Median) vs. Model B (Logit 0.50)\nAgreement Confusion Matrix",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Model B: Logistic Regression Classification (0 = Reject, 1 = Approve)",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Model A: OLS Classification (0 = Reject, 1 = Approve)", fontsize=9, fontweight="bold"
    )

    fig.tight_layout()
    fig.savefig(out_p)
    plt.close(fig)
    return str(out_p)


def plot_simpsons_forest(
    simpsons_res: SimpsonsParadoxResult,
    output_path: str | Path = "reports/figures/simpsons_forest_plot.png",
) -> str:
    """Generate forest plot comparing coefficients and 95% CIs across subgroups."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    table = simpsons_res.comparison_table
    predictors = table["Predictor"].tolist()
    y_pos = np.arange(len(predictors))

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Plot Full model
    b_full = table["Beta_Full"].to_numpy()
    ax.scatter(b_full, y_pos - 0.2, color="#333333", marker="o", s=60, label="Full Cohort (700)")

    # Plot Group A
    b_a = table[f"Beta_{simpsons_res.group_a_name}"].to_numpy()
    ax.scatter(
        b_a,
        y_pos,
        color="#e41a1c",
        marker="s",
        s=60,
        label=f"Subgroup: {simpsons_res.group_a_name} ({simpsons_res.group_a_count})",
    )

    # Plot Group B
    b_b = table[f"Beta_{simpsons_res.group_b_name}"].to_numpy()
    ax.scatter(
        b_b,
        y_pos + 0.2,
        color="#377eb8",
        marker="^",
        s=60,
        label=f"Subgroup: {simpsons_res.group_b_name} ({simpsons_res.group_b_count})",
    )

    ax.axvline(0.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predictors, fontsize=9, fontweight="bold")
    ax.set_xlabel(
        "Standardized Logistic Regression Coefficient (Beta)", fontsize=10, fontweight="bold"
    )
    ax.set_title(
        f"Simpson's Paradox Forest Plot: Stratified by {simpsons_res.subgroup_variable}",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_p)
    plt.close(fig)
    return str(out_p)


def plot_ovb_shifts(
    ovb_res: OmittedVariableBiasResult,
    output_path: str | Path = "reports/figures/ovb_parameter_shifts.png",
    top_n: int = 8,
) -> str:
    """Plot bar chart of largest parameter shifts induced by omitted confounder."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    top_df = ovb_res.comparison_table.head(top_n).copy()
    top_df = top_df.iloc[::-1]  # reverse for horizontal bar chart

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    colors = ["#d62728" if x < 0 else "#2ca02c" for x in top_df["Pct_Change"]]

    bars = ax.barh(
        top_df["Predictor"], top_df["Pct_Change"], color=colors, edgecolor="black", alpha=0.85
    )
    ax.axvline(0.0, color="black", linestyle="-", linewidth=1.0)

    for bar, val in zip(bars, top_df["Pct_Change"]):
        x_pos = val + (4.0 if val >= 0 else -18.0)
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}%",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("Percentage Parameter Shift (% Delta Beta)", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Omitted-Variable Bias: Parameter Shifts from Dropping '{ovb_res.dropped_variable}'",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_p)
    plt.close(fig)
    return str(out_p)


def plot_adversarial_comparison(
    adv_res: AdversarialAuditResult,
    output_path: str | Path = "reports/figures/adversarial_demographic_profiles.png",
) -> str:
    """Generate side-by-side bar chart of adversarial error characteristics vs test cohort."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    table = adv_res.profiling_table.iloc[1:5].copy()  # percentage rows
    labels = table["Metric / Characteristic"].tolist()
    adv_vals = table["Adversarial Subset"].to_numpy()
    test_vals = table["Full Test Set (300)"].to_numpy()

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.bar(
        x - width / 2,
        adv_vals,
        width,
        label=f"Adversarial Subset ({adv_res.total_adversarial_count})",
        color="#d62728",
        edgecolor="black",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        test_vals,
        width,
        label="Full Test Cohort (300)",
        color="#1f77b4",
        edgecolor="black",
        alpha=0.85,
    )

    ax.set_ylabel("Proportion / Percentage (%)", fontsize=10, fontweight="bold")
    ax.set_title(
        "Demographic Profile: High-Confidence Adversarial Errors vs. Test Cohort",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, fontweight="bold", rotation=10)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_p)
    plt.close(fig)
    return str(out_p)


def generate_all_figures(
    dataset: PreprocessedDataset,
    phase1_res: Phase1ComparisonResult,
    simpsons_res: SimpsonsParadoxResult,
    ovb_res: OmittedVariableBiasResult,
    adv_res: AdversarialAuditResult,
    fair_res: FairnessAuditResult,
) -> dict[str, str]:
    """Generate and save all publication-grade figures for reports and notebook."""
    paths: dict[str, str] = {}
    paths["agreement_matrix"] = plot_agreement_confusion_matrix(phase1_res)
    paths["simpsons_forest"] = plot_simpsons_forest(simpsons_res)
    paths["ovb_shifts"] = plot_ovb_shifts(ovb_res)
    paths["adversarial_profiles"] = plot_adversarial_comparison(adv_res)
    if fair_res.figure_path:
        paths["fairness_tradeoff"] = fair_res.figure_path
    return paths
