"""Model implementations and comparative auditing for Phase 1 (Part 1).

Includes Model A (OLS Linear Probability Model with median thresholding) and
Model B (Logistic Regression MLE with 0.50 probability thresholding), agreement
confusion matrix construction, and ground-truth validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.preprocessing import PreprocessedDataset


@dataclass(frozen=True)
class ModelEvaluation:
    """Evaluation metrics for a single model against ground truth."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray  # [[TN, FP], [FN, TP]]
    positive_count: int
    positive_rate: float


@dataclass(frozen=True)
class Phase1ComparisonResult:
    """Comprehensive comparative evaluation between Model A and Model B."""

    ols_model: LinearRegression
    logit_model: LogisticRegression
    y_pred_ols_cont: np.ndarray
    y_pred_logit_prob: np.ndarray
    ols_median: float
    y_pred_ols_bin: np.ndarray
    y_pred_logit_bin: np.ndarray
    agreement_confusion_matrix: pd.DataFrame
    mismatch_count: int
    mismatch_pct: float
    ols_out_of_bounds_neg: int
    ols_out_of_bounds_pos: int
    eval_ols: ModelEvaluation
    eval_logit: ModelEvaluation
    sm_ols_results: Any
    sm_logit_results: Any


def fit_ols_model(
    X_train: np.ndarray,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray,
) -> tuple[LinearRegression, np.ndarray, float, np.ndarray]:
    """Fit Ordinary Least Squares (LPM) and apply test-set median threshold.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (700 rows).
    y_train : np.ndarray | pd.Series
        Training target vector (700 rows).
    X_test : np.ndarray
        Test feature matrix (300 rows).

    Returns
    -------
    tuple[LinearRegression, np.ndarray, float, np.ndarray]
        Fitted model, continuous test predictions, median threshold, binary test predictions.
    """
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    y_pred_cont = ols.predict(X_test)
    median_val = float(np.median(y_pred_cont))
    y_pred_bin = (y_pred_cont >= median_val).astype(int)
    return ols, y_pred_cont, median_val, y_pred_bin


def fit_logit_model(
    X_train: np.ndarray,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray,
    random_state: int = 343,
    max_iter: int = 1000,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """Fit Logistic Regression via MLE and apply 0.50 probability threshold.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (700 rows).
    y_train : np.ndarray | pd.Series
        Training target vector (700 rows).
    X_test : np.ndarray
        Test feature matrix (300 rows).
    random_state : int
        Deterministic random seed.
    max_iter : int
        Maximum iterations for convergence.

    Returns
    -------
    tuple[LogisticRegression, np.ndarray, np.ndarray]
        Fitted model, predicted test probabilities P(Y=1|X), binary test predictions.
    """
    logit = LogisticRegression(random_state=random_state, max_iter=max_iter)
    logit.fit(X_train, y_train)
    y_prob = logit.predict_proba(X_test)[:, 1]
    y_pred_bin = (y_prob >= 0.50).astype(int)
    return logit, y_prob, y_pred_bin


def evaluate_model(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> ModelEvaluation:
    """Compute ground-truth classification metrics for a binary prediction vector.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground-truth labels (0 or 1).
    y_pred : np.ndarray
        Model binary classifications (0 or 1).
    model_name : str
        Label identifier for reporting.

    Returns
    -------
    ModelEvaluation
        Encapsulated metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    pos_count = int(y_pred.sum())
    return ModelEvaluation(
        name=model_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        confusion_matrix=cm,
        positive_count=pos_count,
        positive_rate=float(pos_count / len(y_pred)),
    )


def build_agreement_matrix(
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> pd.DataFrame:
    """Construct 2x2 cross-tabulation of Model A predictions vs Model B predictions.

    Parameters
    ----------
    y_pred_a : np.ndarray
        Binary predictions of Model A (rows).
    y_pred_b : np.ndarray
        Binary predictions of Model B (columns).

    Returns
    -------
    pd.DataFrame
        2x2 contingency matrix with row/column labels and marginal sums.
    """
    df = pd.crosstab(
        pd.Series(y_pred_a, name="Model A (OLS >= median)"),
        pd.Series(y_pred_b, name="Model B (Logit >= 0.50)"),
        margins=True,
        margins_name="Total",
    )
    return df


def run_phase1_analysis(dataset: PreprocessedDataset) -> Phase1ComparisonResult:
    """Execute complete Phase 1 modeling and comparative analysis.

    Parameters
    ----------
    dataset : PreprocessedDataset
        Preprocessed dataset container from Phase 0.

    Returns
    -------
    Phase1ComparisonResult
        Full results comparing Model A and Model B.
    """
    X_tr = dataset.X_train_processed
    y_tr = dataset.y_train.to_numpy()
    X_te = dataset.X_test_processed
    y_te = dataset.y_test.to_numpy()

    # 1. Fit scikit-learn models
    ols, y_ols_cont, ols_med, y_ols_bin = fit_ols_model(X_tr, y_tr, X_te)
    logit, y_logit_prob, y_logit_bin = fit_logit_model(X_tr, y_tr, X_te)

    # 2. Out-of-bounds continuous predictions in OLS
    neg_count = int((y_ols_cont < 0.0).sum())
    pos_count = int((y_ols_cont > 1.0).sum())

    # 3. Model agreement analysis
    agreement_cm = build_agreement_matrix(y_ols_bin, y_logit_bin)
    mismatches = int((y_ols_bin != y_logit_bin).sum())
    mismatch_pct = float(mismatches / len(y_te) * 100.0)

    # 4. Ground-truth evaluations
    eval_ols = evaluate_model(y_te, y_ols_bin, "Model A: OLS (>= median)")
    eval_logit = evaluate_model(y_te, y_logit_bin, "Model B: Logit (>= 0.50)")

    # 5. Fit statsmodels for robust inference tables
    X_tr_sm = sm.add_constant(X_tr, has_constant="add")
    sm_ols = sm.OLS(y_tr, X_tr_sm).fit()
    try:
        sm_logit = sm.Logit(y_tr, X_tr_sm).fit(disp=False, maxiter=500)
    except Exception:
        sm_logit = None

    return Phase1ComparisonResult(
        ols_model=ols,
        logit_model=logit,
        y_pred_ols_cont=y_ols_cont,
        y_pred_logit_prob=y_logit_prob,
        ols_median=ols_med,
        y_pred_ols_bin=y_ols_bin,
        y_pred_logit_bin=y_logit_bin,
        agreement_confusion_matrix=agreement_cm,
        mismatch_count=mismatches,
        mismatch_pct=mismatch_pct,
        ols_out_of_bounds_neg=neg_count,
        ols_out_of_bounds_pos=pos_count,
        eval_ols=eval_ols,
        eval_logit=eval_logit,
        sm_ols_results=sm_ols,
        sm_logit_results=sm_logit,
    )
