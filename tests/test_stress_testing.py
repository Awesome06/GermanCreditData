"""Unit tests for Phase 2 stress-testing and the Four Traps."""

from pathlib import Path

import numpy as np

from src.models import run_phase1_analysis
from src.preprocessing import prepare_preprocessed_dataset
from src.stress_testing import (
    audit_adversarial_subset,
    audit_fairness_threshold_sweep,
    audit_omitted_variable_bias,
    audit_simpsons_paradox,
)


def test_audit_simpsons_paradox() -> None:
    dataset = prepare_preprocessed_dataset()
    res = audit_simpsons_paradox(dataset, split_by="sex")
    assert len(res.comparison_table) >= 5
    assert len(res.sign_reversals) > 0 or len(res.divergent_predictors) > 0
    assert res.group_a_count + res.group_b_count == 700


def test_audit_omitted_variable_bias() -> None:
    dataset = prepare_preprocessed_dataset()
    res = audit_omitted_variable_bias(dataset, drop_confounder="checking_status")
    assert len(res.comparison_table) > 30
    assert abs(res.max_pct_change) > 50.0  # Confirms extreme parameter volatility


def test_audit_adversarial_subset() -> None:
    dataset = prepare_preprocessed_dataset()
    phase1_res = run_phase1_analysis(dataset)
    adv_res = audit_adversarial_subset(dataset, phase1_res)

    assert adv_res.total_adversarial_count == 26
    assert adv_res.fp_count == 22
    assert adv_res.fn_count == 4
    np.testing.assert_allclose(adv_res.adversarial_pct, 8.6666, atol=1e-2)
    assert len(adv_res.profiling_table) >= 6


def test_audit_fairness_threshold_sweep() -> None:
    dataset = prepare_preprocessed_dataset()
    phase1_res = run_phase1_analysis(dataset)
    out_fig = "reports/figures/test_fairness_chart.png"
    fair_res = audit_fairness_threshold_sweep(
        dataset, phase1_res, sensitive_var="age", output_figure_path=out_fig
    )

    assert len(fair_res.metrics_table) == 10
    assert fair_res.tradeoff_cost > 0.0
    assert Path(out_fig).is_file()

    # Clean up test artifact
    Path(out_fig).unlink(missing_ok=True)
