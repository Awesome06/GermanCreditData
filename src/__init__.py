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

__all__ = [
    "GLOBAL_RANDOM_SEED",
    "PreprocessedDataset",
    "ModelEvaluation",
    "Phase1ComparisonResult",
    "audit_missing_and_structural_absence",
    "audit_outliers",
    "build_agreement_matrix",
    "derive_sensitive_attributes",
    "evaluate_model",
    "fit_logit_model",
    "fit_ols_model",
    "load_raw_data",
    "prepare_preprocessed_dataset",
    "reencode_target",
    "run_phase1_analysis",
]
