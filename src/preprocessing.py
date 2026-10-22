"""Data preprocessing and feature engineering pipeline for UCI German Credit Data.

Enforces deterministic random seed locking (343), 700/300 train-test partitioning,
structural absence level preservation, and strict zero-data-leakage scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GLOBAL_RANDOM_SEED: Final[int] = 343

FEATURE_COLUMNS: Final[list[str]] = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "maintenance_people",
    "telephone",
    "foreign_worker",
    "target",
]

NUMERICAL_COLUMNS: Final[list[str]] = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age_years",
    "existing_credits",
    "maintenance_people",
]

CATEGORICAL_COLUMNS: Final[list[str]] = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]


@dataclass(frozen=True)
class PreprocessedDataset:
    """Container holding partitioned and processed training/test matrices with metadata."""

    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_processed: np.ndarray
    X_test_processed: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler
    encoder: OneHotEncoder
    sensitive_train: pd.DataFrame
    sensitive_test: pd.DataFrame


def load_raw_data(data_path: str | Path = "data/german.data") -> pd.DataFrame:
    """Load raw german.data and assign standard column headers.

    Parameters
    ----------
    data_path : str | Path
        Path to space-delimited german.data file.

    Returns
    -------
    pd.DataFrame
        Loaded 1000-row dataframe with 21 columns.
    """
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"German Credit dataset file not found at: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, names=FEATURE_COLUMNS)
    if df.shape != (1000, 21):
        raise ValueError(f"Expected shape (1000, 21), got {df.shape}")
    return df


def reencode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Re-encode UCI target so Approve = 1 (formerly 1) and Reject = 0 (formerly 2).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'target' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with transformed binary target.
    """
    df_copy = df.copy()
    target_mapping = {1: 1, 2: 0}
    df_copy["target"] = df_copy["target"].map(target_mapping)
    if not set(df_copy["target"].unique()).issubset({0, 1}):
        raise ValueError("Target mapping produced invalid values outside {0, 1}.")
    return df_copy


def derive_sensitive_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Derive standardized binary demographic indicators for fairness stress tests.

    - Age Group: Young (< 25) = 1, Older (>= 25) = 0
    - Sex: Female = 1 (A92, A95), Male = 0 (A91, A93, A94)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'age_years' and 'personal_status_sex'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 'age_young' (1/0) and 'sex_female' (1/0).
    """
    sensitive_df = pd.DataFrame(index=df.index)
    sensitive_df["age_young"] = (df["age_years"] < 25).astype(int)
    female_codes = {"A92", "A95"}
    sensitive_df["sex_female"] = df["personal_status_sex"].isin(female_codes).astype(int)
    sensitive_df["age_years"] = df["age_years"]
    sensitive_df["personal_status_sex"] = df["personal_status_sex"]
    return sensitive_df


def audit_missing_and_structural_absence(df: pd.DataFrame) -> dict[str, int]:
    """Audit missing values and counts of structural absence codes.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or preprocessed dataset.

    Returns
    -------
    dict[str, int]
        Audit summary mapping code to count.
    """
    return {
        "null_count_total": int(df.isnull().sum().sum()),
        "checking_status_A14_no_account": int((df["checking_status"] == "A14").sum()),
        "savings_status_A65_no_account": int((df["savings_status"] == "A65").sum()),
    }


def audit_outliers(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute summary statistics, skewness, and IQR bounds for continuous financial variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing numerical predictors.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dictionary of outlier metrics per variable.
    """
    outlier_metrics: dict[str, dict[str, float]] = {}
    for col in ["credit_amount", "duration_months", "age_years"]:
        series = df[col]
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())
        outlier_metrics[col] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "skewness": float(series.skew()),
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": float(outlier_count),
            "outlier_pct": float(outlier_count / len(df) * 100.0),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return outlier_metrics


def prepare_preprocessed_dataset(
    data_path: str | Path = "data/german.data",
    random_state: int = GLOBAL_RANDOM_SEED,
    stratify: bool = True,
) -> PreprocessedDataset:
    """Execute complete 4-pillar data preprocessing pipeline with zero data leakage.

    1. Ingestion & Target Re-encoding (Approve=1, Reject=0)
    2. Sensitive attribute extraction (Age < 25, Sex)
    3. Deterministic 70/30 train-test partition (700 train / 300 test)
    4. StandardScaler fitted strictly on training numerical features
    5. OneHotEncoder (drop='first') fitted strictly on training categorical features

    Parameters
    ----------
    data_path : str | Path
        Path to raw german.data file.
    random_state : int
        Deterministic random seed (default: 343).
    stratify : bool
        Whether to stratify split by target (default: True).

    Returns
    -------
    PreprocessedDataset
        Encapsulated preprocessed dataset ready for modeling.
    """
    raw_df = load_raw_data(data_path)
    clean_df = reencode_target(raw_df)
    sensitive_all = derive_sensitive_attributes(clean_df)

    X = clean_df.drop(columns=["target"])
    y = clean_df["target"]

    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X,
        y,
        sensitive_all,
        test_size=0.30,
        random_state=random_state,
        stratify=stratify_target,
    )

    if len(X_train) != 700 or len(X_test) != 300:
        raise ValueError(
            f"Expected partition sizes 700 and 300, got {len(X_train)} and {len(X_test)}."
        )

    # 1. Scale numerical features (fitted strictly on X_train to prevent leakage)
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train[NUMERICAL_COLUMNS])
    X_test_num_scaled = scaler.transform(X_test[NUMERICAL_COLUMNS])

    # 2. One-hot encode categorical features (drop='first' to prevent dummy variable trap)
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    X_train_cat_ohe = encoder.fit_transform(X_train[CATEGORICAL_COLUMNS])
    X_test_cat_ohe = encoder.transform(X_test[CATEGORICAL_COLUMNS])

    cat_feature_names = list(encoder.get_feature_names_out(CATEGORICAL_COLUMNS))
    full_feature_names = NUMERICAL_COLUMNS + cat_feature_names

    X_train_proc = np.hstack([X_train_num_scaled, X_train_cat_ohe])
    X_test_proc = np.hstack([X_test_num_scaled, X_test_cat_ohe])

    return PreprocessedDataset(
        X_train_raw=X_train,
        X_test_raw=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_processed=X_train_proc,
        X_test_processed=X_test_proc,
        feature_names=full_feature_names,
        scaler=scaler,
        encoder=encoder,
        sensitive_train=sens_train,
        sensitive_test=sens_test,
    )
