"""German Credit Data statistical auditing and algorithmic fairness evaluation package."""

from src.models import (
    ModelEvaluation,
    Phase1ComparisonResult,
    build_agreement_matrix,
    evaluate_model,
    fit_logit_model,
    fit_ols_model,
    run_phase1_analysis,
)
from src.preprocessing import (
    GLOBAL_RANDOM_SEED,
    PreprocessedDataset,
    audit_missing_and_structural_absence,
    audit_outliers,
    derive_sensitive_attributes,
    load_raw_data,
    prepare_preprocessed_dataset,
    reencode_target,
)
from src.stress_testing import (
    AdversarialAuditResult,
    FairnessAuditResult,
    OmittedVariableBiasResult,
    SimpsonsParadoxResult,
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

__all__ = [
    "GLOBAL_RANDOM_SEED",
    "PreprocessedDataset",
    "ModelEvaluation",
    "Phase1ComparisonResult",
    "SimpsonsParadoxResult",
    "OmittedVariableBiasResult",
    "AdversarialAuditResult",
    "FairnessAuditResult",
    "audit_adversarial_subset",
    "audit_fairness_threshold_sweep",
    "audit_missing_and_structural_absence",
    "audit_omitted_variable_bias",
    "audit_outliers",
    "audit_simpsons_paradox",
    "build_agreement_matrix",
    "derive_sensitive_attributes",
    "evaluate_model",
    "fit_logit_model",
    "fit_ols_model",
    "generate_all_figures",
    "load_raw_data",
    "plot_adversarial_comparison",
    "plot_agreement_confusion_matrix",
    "plot_ovb_shifts",
    "plot_simpsons_forest",
    "prepare_preprocessed_dataset",
    "reencode_target",
    "run_phase1_analysis",
]
