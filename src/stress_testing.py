"""Phase 2: Stress-testing and the Four Traps auditing module.

Implements:
2.1 Simpson's Paradox Hunt (Subgroup stratification, sign reversals, 95% CIs)
2.2 Omitted-Variable Bias (Confounder drop and parameter instability quantification)
2.3 Adversarial Subset Construction (High-confidence failure mode isolation)
2.4 Multi-Threshold Algorithmic Fairness Auditing (DP, EO, PP sweep and trade-off cost)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from src.models import Phase1ComparisonResult
from src.preprocessing import PreprocessedDataset


@dataclass(frozen=True)
class SimpsonsParadoxResult:
    """Results from Simpson's paradox demographic subgroup auditing."""

    subgroup_variable: str
    group_a_name: str
    group_b_name: str
    group_a_count: int
    group_b_count: int
    comparison_table: pd.DataFrame
    sign_reversals: list[str]
    divergent_predictors: list[str]
    sm_full: Any
    sm_subgroup_a: Any
    sm_subgroup_b: Any


@dataclass(frozen=True)
class OmittedVariableBiasResult:
    """Results from omitted-variable bias sensitivity analysis."""

    dropped_variable: str
    comparison_table: pd.DataFrame
    max_shifted_feature: str
    max_pct_change: float


@dataclass(frozen=True)
class AdversarialAuditResult:
    """Results from high-confidence adversarial error profiling."""

    fp_count: int
    fn_count: int
    total_adversarial_count: int
    adversarial_pct: float
    adversarial_indices: np.ndarray
    profiling_table: pd.DataFrame
    dominant_shared_traits: dict[str, float]


@dataclass(frozen=True)
class FairnessAuditResult:
    """Results from 10-threshold fairness trade-off sweep."""

    sensitive_attribute: str
    group_a_label: str
    group_b_label: str
    metrics_table: pd.DataFrame
    tradeoff_cost: float
    tradeoff_summary: str
    figure_path: str | None


def audit_simpsons_paradox(
    dataset: PreprocessedDataset,
    split_by: str = "sex",
) -> SimpsonsParadoxResult:
    """Perform Simpson's paradox audit comparing Full model vs Subgroup A vs Subgroup B.

    Parameters
    ----------
    dataset : PreprocessedDataset
        Preprocessed training/test datasets.
    split_by : str
        Demographic attribute ('sex' or 'age'). Default is 'sex'.

    Returns
    -------
    SimpsonsParadoxResult
        Audit tables, sign reversals, and confidence intervals.
    """
    df_train = dataset.X_train_raw.copy()
    y_train = dataset.y_train.to_numpy()

    cols_to_evaluate = [
        "duration_months",
        "credit_amount",
        "installment_rate",
        "residence_since",
        "age_years",
        "existing_credits",
        "maintenance_people",
    ]

    # Standardize numericals for direct coefficient comparability
    df_std = df_train.copy()
    for col in cols_to_evaluate:
        std_val = df_train[col].std()
        df_std[col] = (df_train[col] - df_train[col].mean()) / (std_val if std_val > 0 else 1.0)

    if split_by.lower() == "age":
        var_name = "Age Group"
        name_a, name_b = "Young (<25)", "Older (>=25)"
        mask_a = (df_train["age_years"] < 25).to_numpy()
        mask_b = (df_train["age_years"] >= 25).to_numpy()
        predictors = [c for c in cols_to_evaluate if c != "age_years"]
    else:
        var_name = "Sex / Marital Status"
        name_a, name_b = "Female", "Male"
        mask_a = df_train["personal_status_sex"].isin(["A92", "A95"]).to_numpy()
        mask_b = df_train["personal_status_sex"].isin(["A91", "A93", "A94"]).to_numpy()
        predictors = cols_to_evaluate

    X_full = sm.add_constant(df_std[predictors], has_constant="add")
    X_a = sm.add_constant(df_std[predictors].iloc[mask_a], has_constant="add")
    X_b = sm.add_constant(df_std[predictors].iloc[mask_b], has_constant="add")

    sm_full = sm.Logit(y_train, X_full).fit(disp=False, maxiter=500)
    sm_a = sm.Logit(y_train[mask_a], X_a).fit(disp=False, maxiter=500)
    sm_b = sm.Logit(y_train[mask_b], X_b).fit(disp=False, maxiter=500)

    ci_full = sm_full.conf_int()
    ci_a = sm_a.conf_int()
    ci_b = sm_b.conf_int()

    rows = []
    sign_reversals = []
    divergent_predictors = []

    for pred in predictors:
        b_f = float(sm_full.params[pred])
        p_f = float(sm_full.pvalues[pred])
        ci_f_low, ci_f_high = float(ci_full.loc[pred, 0]), float(ci_full.loc[pred, 1])

        b_a = float(sm_a.params[pred])
        p_a = float(sm_a.pvalues[pred])
        ci_a_low, ci_a_high = float(ci_a.loc[pred, 0]), float(ci_a.loc[pred, 1])

        b_b = float(sm_b.params[pred])
        p_b = float(sm_b.pvalues[pred])
        ci_b_low, ci_b_high = float(ci_b.loc[pred, 0]), float(ci_b.loc[pred, 1])

        # Detect sign reversals between full and subgroup or between subgroups
        if (b_a * b_b < 0) or (b_f * b_a < 0) or (b_f * b_b < 0):
            sign_reversals.append(pred)

        # Detect magnitude/significance divergence (significant in one, insignificant in other)
        if (p_a < 0.05 and p_b >= 0.10) or (p_b < 0.05 and p_a >= 0.10):
            divergent_predictors.append(pred)

        rows.append(
            {
                "Predictor": pred,
                "Beta_Full": b_f,
                "p_Full": p_f,
                "CI_Full": f"[{ci_f_low:.2f}, {ci_f_high:.2f}]",
                f"Beta_{name_a}": b_a,
                f"p_{name_a}": p_a,
                f"CI_{name_a}": f"[{ci_a_low:.2f}, {ci_a_high:.2f}]",
                f"Beta_{name_b}": b_b,
                f"p_{name_b}": p_b,
                f"CI_{name_b}": f"[{ci_b_low:.2f}, {ci_b_high:.2f}]",
            }
        )

    comp_df = pd.DataFrame(rows)

    return SimpsonsParadoxResult(
        subgroup_variable=var_name,
        group_a_name=name_a,
        group_b_name=name_b,
        group_a_count=int(mask_a.sum()),
        group_b_count=int(mask_b.sum()),
        comparison_table=comp_df,
        sign_reversals=list(set(sign_reversals)),
        divergent_predictors=list(set(divergent_predictors)),
        sm_full=sm_full,
        sm_subgroup_a=sm_a,
        sm_subgroup_b=sm_b,
    )


def audit_omitted_variable_bias(
    dataset: PreprocessedDataset,
    drop_confounder: str = "checking_status",
) -> OmittedVariableBiasResult:
    """Quantify coefficient volatility and OVB shifts when omitting a key confounder.

    Parameters
    ----------
    dataset : PreprocessedDataset
        Preprocessed training/test matrices.
    drop_confounder : str
        Variable prefix or name to drop ('checking_status' or 'credit_amount').

    Returns
    -------
    OmittedVariableBiasResult
        Parameter shift table with absolute and percentage shifts.
    """
    X_tr = dataset.X_train_processed
    y_tr = dataset.y_train.to_numpy()
    feature_names = dataset.feature_names

    # Fit Full Model
    clf_full = LogisticRegression(random_state=343, max_iter=1000, C=1e5)
    clf_full.fit(X_tr, y_tr)
    full_series = pd.Series(clf_full.coef_[0], index=feature_names)

    # Determine remaining features
    if drop_confounder == "checking_status":
        keep_indices = [
            i for i, name in enumerate(feature_names) if not name.startswith("checking_status")
        ]
    elif drop_confounder == "credit_amount":
        keep_indices = [i for i, name in enumerate(feature_names) if name != "credit_amount"]
    else:
        keep_indices = [
            i for i, name in enumerate(feature_names) if not name.startswith(drop_confounder)
        ]

    X_tr_red = X_tr[:, keep_indices]
    reduced_names = [feature_names[i] for i in keep_indices]

    clf_reduced = LogisticRegression(random_state=343, max_iter=1000, C=1e5)
    clf_reduced.fit(X_tr_red, y_tr)
    reduced_series = pd.Series(clf_reduced.coef_[0], index=reduced_names)

    shift_df = pd.DataFrame(
        {
            "Predictor": reduced_names,
            "Beta_Full": full_series[reduced_names].to_numpy(),
            "Beta_Reduced": reduced_series.to_numpy(),
        }
    )
    shift_df["Delta_Beta"] = shift_df["Beta_Reduced"] - shift_df["Beta_Full"]
    shift_df["Pct_Change"] = (shift_df["Delta_Beta"] / shift_df["Beta_Full"].abs()) * 100.0

    sorted_df = shift_df.sort_values(by="Pct_Change", key=abs, ascending=False).reset_index(
        drop=True
    )
    max_feature = str(sorted_df.iloc[0]["Predictor"])
    max_pct = float(sorted_df.iloc[0]["Pct_Change"])

    return OmittedVariableBiasResult(
        dropped_variable=drop_confounder,
        comparison_table=sorted_df,
        max_shifted_feature=max_feature,
        max_pct_change=max_pct,
    )


def audit_adversarial_subset(
    dataset: PreprocessedDataset,
    phase1_result: Phase1ComparisonResult,
    high_conf_pos_threshold: float = 0.75,
    high_conf_neg_threshold: float = 0.25,
    max_sample: int = 50,
) -> AdversarialAuditResult:
    """Extract and profile high-confidence incorrect predictions on the 300-row test set.

    High-confidence False Positives: P(Approve) >= 0.75 and actual Y = 0 (Default)
    High-confidence False Negatives: P(Approve) <= 0.25 and actual Y = 1 (Repaid)

    Parameters
    ----------
    dataset : PreprocessedDataset
        Preprocessed dataset.
    phase1_result : Phase1ComparisonResult
        Fitted model probabilities on test set.
    high_conf_pos_threshold : float
        Threshold for high-confidence false positive (default: 0.75).
    high_conf_neg_threshold : float
        Threshold for high-confidence false negative (default: 0.25).
    max_sample : int
        Maximum sample size if more than max_sample exist (default: 50).

    Returns
    -------
    AdversarialAuditResult
        Audit metrics and comparative demographic profile table.
    """
    y_prob = phase1_result.y_pred_logit_prob
    y_test = dataset.y_test.to_numpy()

    idx_fp = np.where((y_prob >= high_conf_pos_threshold) & (y_test == 0))[0]
    idx_fn = np.where((y_prob <= high_conf_neg_threshold) & (y_test == 1))[0]

    fp_count = len(idx_fp)
    fn_count = len(idx_fn)
    all_adv_idx = np.concatenate([idx_fp, idx_fn])
    total_adv = len(all_adv_idx)

    # Sample if > max_sample, else use all
    if total_adv > max_sample:
        rng = np.random.default_rng(343)
        selected_idx = rng.choice(all_adv_idx, size=max_sample, replace=False)
    else:
        selected_idx = all_adv_idx

    df_test_raw = dataset.X_test_raw.copy()
    df_adv = df_test_raw.iloc[selected_idx].copy()

    # Comparative demographic and feature profiling
    profiles = [
        ("Total Applicants", float(len(df_adv)), float(len(df_test_raw))),
        (
            "Young Applicants (<25)",
            float((df_adv["age_years"] < 25).mean() * 100),
            float((df_test_raw["age_years"] < 25).mean() * 100),
        ),
        (
            "Female Applicants",
            float(df_adv["personal_status_sex"].isin(["A92", "A95"]).mean() * 100),
            float(df_test_raw["personal_status_sex"].isin(["A92", "A95"]).mean() * 100),
        ),
        (
            "Foreign Workers (A201)",
            float((df_adv["foreign_worker"] == "A201").mean() * 100),
            float((df_test_raw["foreign_worker"] == "A201").mean() * 100),
        ),
        (
            "No Checking Account (A14)",
            float((df_adv["checking_status"] == "A14").mean() * 100),
            float((df_test_raw["checking_status"] == "A14").mean() * 100),
        ),
        (
            "Mean Credit Amount (DM)",
            float(df_adv["credit_amount"].mean()),
            float(df_test_raw["credit_amount"].mean()),
        ),
        (
            "Mean Duration (Months)",
            float(df_adv["duration_months"].mean()),
            float(df_test_raw["duration_months"].mean()),
        ),
    ]

    prof_df = pd.DataFrame(
        profiles, columns=["Metric / Characteristic", "Adversarial Subset", "Full Test Set (300)"]
    )

    dominant_traits = {
        "foreign_worker_pct": float((df_adv["foreign_worker"] == "A201").mean() * 100),
        "no_checking_account_pct": float((df_adv["checking_status"] == "A14").mean() * 100),
        "mean_credit_amount": float(df_adv["credit_amount"].mean()),
    }

    return AdversarialAuditResult(
        fp_count=fp_count,
        fn_count=fn_count,
        total_adversarial_count=total_adv,
        adversarial_pct=float(total_adv / len(y_test) * 100.0),
        adversarial_indices=selected_idx,
        profiling_table=prof_df,
        dominant_shared_traits=dominant_traits,
    )


def audit_fairness_threshold_sweep(
    dataset: PreprocessedDataset,
    phase1_result: Phase1ComparisonResult,
    sensitive_var: str = "age",
    output_figure_path: str | Path = "reports/figures/fairness_tradeoff_chart.png",
) -> FairnessAuditResult:
    """Sweep 10 decision thresholds and calculate DP, EO, PP, and trade-off ratios.

    Parameters
    ----------
    dataset : PreprocessedDataset
        Preprocessed test data with sensitive labels.
    phase1_result : Phase1ComparisonResult
        Model test probabilities.
    sensitive_var : str
        Attribute to audit ('age' for Young <25 or 'sex' for Female).
    output_figure_path : str | Path
        Path to save 2D trade-off chart.

    Returns
    -------
    FairnessAuditResult
        10-row metrics table, trade-off cost, and plot path.
    """
    y_prob = phase1_result.y_pred_logit_prob
    y_test = dataset.y_test.to_numpy()
    sens_test = dataset.sensitive_test

    if sensitive_var.lower() == "sex":
        attr_name = "Sex"
        group_a_label = "Female"
        group_b_label = "Male"
        mask_a = (sens_test["sex_female"] == 1).to_numpy()
        mask_b = (sens_test["sex_female"] == 0).to_numpy()
    else:
        attr_name = "Age Group"
        group_a_label = "Young (<25)"
        group_b_label = "Older (>=25)"
        mask_a = (sens_test["age_young"] == 1).to_numpy()
        mask_b = (sens_test["age_young"] == 0).to_numpy()

    thresholds = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    rows = []

    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)

        # Approval Rates
        p_hat_a = float(y_hat[mask_a].mean())
        p_hat_b = float(y_hat[mask_b].mean())
        dp = float(p_hat_a / p_hat_b) if p_hat_b > 0 else np.nan

        # Error Rates (Group A)
        y_a, y_hat_a = y_test[mask_a], y_hat[mask_a]
        tp_a = int(((y_hat_a == 1) & (y_a == 1)).sum())
        pos_a = int((y_a == 1).sum())
        fp_a = int(((y_hat_a == 1) & (y_a == 0)).sum())
        neg_a = int((y_a == 0).sum())
        tpr_a = float(tp_a / pos_a) if pos_a > 0 else 0.0
        fpr_a = float(fp_a / neg_a) if neg_a > 0 else 0.0

        # Error Rates (Group B)
        y_b, y_hat_b = y_test[mask_b], y_hat[mask_b]
        tp_b = int(((y_hat_b == 1) & (y_b == 1)).sum())
        pos_b = int((y_b == 1).sum())
        fp_b = int(((y_hat_b == 1) & (y_b == 0)).sum())
        neg_b = int((y_b == 0).sum())
        tpr_b = float(tp_b / pos_b) if pos_b > 0 else 0.0
        fpr_b = float(fp_b / neg_b) if neg_b > 0 else 0.0

        # Equalized Odds ratio
        ratio_tpr = float(tpr_a / tpr_b) if tpr_b > 0 else (1.0 if tpr_a == 0 else np.nan)
        ratio_fpr = float(fpr_a / fpr_b) if fpr_b > 0 else (1.0 if fpr_a == 0 else np.nan)
        eo = (
            float(min(ratio_tpr, ratio_fpr))
            if not (np.isnan(ratio_tpr) or np.isnan(ratio_fpr))
            else np.nan
        )

        # Predictive Parity (Precision Ratio)
        prec_a = float(tp_a / y_hat_a.sum()) if y_hat_a.sum() > 0 else np.nan
        prec_b = float(tp_b / y_hat_b.sum()) if y_hat_b.sum() > 0 else np.nan
        pp = float(prec_a / prec_b) if (prec_b and prec_b > 0) else np.nan

        acc = float((y_hat == y_test).mean())

        rows.append(
            {
                "Threshold": t,
                "Demographic_Parity": dp,
                "Equalized_Odds": eo,
                "Predictive_Parity": pp,
                "Overall_Accuracy": acc,
                "TPR_Group_A": tpr_a,
                "TPR_Group_B": tpr_b,
                "FPR_Group_A": fpr_a,
                "FPR_Group_B": fpr_b,
                "Approval_Rate_A": p_hat_a,
                "Approval_Rate_B": p_hat_b,
            }
        )

    metrics_df = pd.DataFrame(rows)

    # Calculate Trade-Off Cost W/Z between two candidate thresholds (e.g., t=0.35 vs t=0.55)
    row_t1 = metrics_df[metrics_df["Threshold"] == 0.35].iloc[0]
    row_t2 = metrics_df[metrics_df["Threshold"] == 0.55].iloc[0]

    dp_gain = float(
        (row_t1["Demographic_Parity"] - row_t2["Demographic_Parity"])
        / row_t2["Demographic_Parity"]
        * 100.0
    )
    eo_loss = float(
        (row_t2["Equalized_Odds"] - row_t1["Equalized_Odds"]) / row_t2["Equalized_Odds"] * 100.0
    )
    tradeoff_cost = float(abs(eo_loss / dp_gain)) if abs(dp_gain) > 0 else 0.0

    dp1 = row_t1["Demographic_Parity"]
    dp2 = row_t2["Demographic_Parity"]
    eo1 = row_t1["Equalized_Odds"]
    eo2 = row_t2["Equalized_Odds"]

    tradeoff_summary = (
        f"Shifting threshold from t=0.55 to t=0.35 improves Demographic Parity from "
        f"{dp2:.4f} to {dp1:.4f} (a {dp_gain:+.2f}% gain), "
        f"while Equalized Odds shifts from {eo2:.4f} to {eo1:.4f} "
        f"({eo_loss:+.2f}% relative change). Trade-Off Cost (W/Z): {tradeoff_cost:.4f}."
    )

    # Generate publication-grade 2D Trade-Off Chart
    fig_path: str | None = None
    if output_figure_path:
        out_p = Path(output_figure_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        valid_points = metrics_df.dropna(subset=["Demographic_Parity", "Equalized_Odds"])

        ax.scatter(
            valid_points["Demographic_Parity"],
            valid_points["Equalized_Odds"],
            color="#1f77b4",
            s=80,
            edgecolors="black",
            zorder=4,
        )

        for _, r in valid_points.iterrows():
            ax.annotate(
                f"t={r['Threshold']:.2f}",
                (r["Demographic_Parity"], r["Equalized_Odds"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                weight="bold",
            )

        ax.axvline(1.0, color="#d62728", linestyle="--", alpha=0.7, label="Ideal DP (1.0)")
        ax.axhline(1.0, color="#2ca02c", linestyle="--", alpha=0.7, label="Ideal EO (1.0)")

        ax.set_xlim(0.0, 1.5)
        ax.set_ylim(0.0, 1.5)
        ax.set_xlabel(
            "Demographic Parity Ratio: P(Ŷ=1|Group A) / P(Ŷ=1|Group B)",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_ylabel(
            "Equalized Odds Ratio: min(TPR ratio, FPR ratio)", fontsize=10, fontweight="bold"
        )
        ax.set_title(
            f"Fairness Trade-Off Frontier across Thresholds: {attr_name}",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(out_p)
        plt.close(fig)
        fig_path = str(out_p)

    return FairnessAuditResult(
        sensitive_attribute=attr_name,
        group_a_label=group_a_label,
        group_b_label=group_b_label,
        metrics_table=metrics_df,
        tradeoff_cost=tradeoff_cost,
        tradeoff_summary=tradeoff_summary,
        figure_path=fig_path,
    )
