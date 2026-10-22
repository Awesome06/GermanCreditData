"""Unit tests for visualization generation pipeline."""

from pathlib import Path

from src.models import run_phase1_analysis
from src.preprocessing import prepare_preprocessed_dataset
from src.stress_testing import (
    audit_adversarial_subset,
    audit_fairness_threshold_sweep,
    audit_omitted_variable_bias,
    audit_simpsons_paradox,
)
from src.visualizations import (
    generate_all_figures,
    plot_adversarial_comparison,
    plot_agreement_confusion_matrix,
    plot_ovb_shifts,
    plot_simpsons_forest,
)


def test_visualizations_generation() -> None:
    dataset = prepare_preprocessed_dataset()
    phase1 = run_phase1_analysis(dataset)
    simpsons = audit_simpsons_paradox(dataset)
    ovb = audit_omitted_variable_bias(dataset)
    adv = audit_adversarial_subset(dataset, phase1)
    fair = audit_fairness_threshold_sweep(
        dataset, phase1, sensitive_var="age", output_figure_path="reports/figures/test_fair.png"
    )

    p1 = plot_agreement_confusion_matrix(phase1, "reports/figures/test_cm.png")
    p2 = plot_simpsons_forest(simpsons, "reports/figures/test_forest.png")
    p3 = plot_ovb_shifts(ovb, "reports/figures/test_ovb.png")
    p4 = plot_adversarial_comparison(adv, "reports/figures/test_adv.png")
    paths = generate_all_figures(dataset, phase1, simpsons, ovb, adv, fair)

    assert Path(p1).is_file()
    assert Path(p2).is_file()
    assert Path(p3).is_file()
    assert Path(p4).is_file()
    assert "fairness_tradeoff" in paths
    assert Path(paths["fairness_tradeoff"]).is_file()

    # Clean up test artifacts
    for p in [p1, p2, p3, p4, "reports/figures/test_fair.png"]:
        Path(p).unlink(missing_ok=True)
