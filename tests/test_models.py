"""Unit tests for Phase 1 modeling and comparative analysis."""

import numpy as np

from src.models import (
    build_agreement_matrix,
    evaluate_model,
    fit_logit_model,
    fit_ols_model,
    run_phase1_analysis,
)
from src.preprocessing import prepare_preprocessed_dataset


def test_ols_and_logit_fitting() -> None:
    dataset = prepare_preprocessed_dataset()
    X_tr = dataset.X_train_processed
    y_tr = dataset.y_train.to_numpy()
    X_te = dataset.X_test_processed

    # Test OLS
    ols, y_ols_cont, ols_med, y_ols_bin = fit_ols_model(X_tr, y_tr, X_te)
    assert len(y_ols_cont) == 300
    assert len(y_ols_bin) == 300
    assert y_ols_bin.sum() == 150  # Median split must force exactly 50% positive
    assert 0.70 <= ols_med <= 0.75

    # Test Logit
    logit, y_prob, y_logit_bin = fit_logit_model(X_tr, y_tr, X_te)
    assert len(y_prob) == 300
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))
    assert y_logit_bin.sum() > 200  # Approves ~75-80% due to 70% base rate


def test_agreement_matrix_and_metrics() -> None:
    y_a = np.array([0, 1, 0, 1, 1])
    y_b = np.array([0, 1, 1, 1, 0])
    cm = build_agreement_matrix(y_a, y_b)
    assert cm.loc["Total", "Total"] == 5

    eval_res = evaluate_model(y_a, y_b, "Test Model")
    assert 0.0 <= eval_res.accuracy <= 1.0


def test_run_phase1_analysis_exact_benchmarks() -> None:
    dataset = prepare_preprocessed_dataset()
    res = run_phase1_analysis(dataset)

    # 1. Exact mismatch count and percentage
    assert res.mismatch_count == 82
    np.testing.assert_allclose(res.mismatch_pct, 27.3333, atol=1e-2)

    # 2. OLS Median value
    np.testing.assert_allclose(res.ols_median, 0.7201, atol=1e-3)

    # 3. Model positive counts
    assert res.eval_ols.positive_count == 150
    assert res.eval_logit.positive_count == 232

    # 4. Out-of-bounds predictions in OLS
    assert res.ols_out_of_bounds_neg == 3
    assert res.ols_out_of_bounds_pos == 39

    # 5. Ground truth accuracy
    np.testing.assert_allclose(res.eval_ols.accuracy, 0.6600, atol=1e-3)
    np.testing.assert_allclose(res.eval_logit.accuracy, 0.7467, atol=1e-3)
