import numpy as np

from src.preprocessing import (
    GLOBAL_RANDOM_SEED,
    NUMERICAL_COLUMNS,
    audit_missing_and_structural_absence,
    audit_outliers,
    derive_sensitive_attributes,
    load_raw_data,
    prepare_preprocessed_dataset,
    reencode_target,
)


def test_load_raw_data() -> None:
    df = load_raw_data()
    assert df.shape == (1000, 21)
    assert "target" in df.columns


def test_reencode_target() -> None:
    raw_df = load_raw_data()
    clean_df = reencode_target(raw_df)
    counts = clean_df["target"].value_counts().to_dict()
    assert counts[1] == 700  # Approve
    assert counts[0] == 300  # Reject


def test_sensitive_attributes() -> None:
    df = load_raw_data()
    sensitive = derive_sensitive_attributes(df)
    assert sensitive["age_young"].sum() == 149
    assert sensitive["sex_female"].sum() == 310


def test_structural_absence_audit() -> None:
    df = load_raw_data()
    audit = audit_missing_and_structural_absence(df)
    assert audit["null_count_total"] == 0
    assert audit["checking_status_A14_no_account"] == 394
    assert audit["savings_status_A65_no_account"] == 183


def test_outlier_audit() -> None:
    df = load_raw_data()
    outliers = audit_outliers(df)
    assert "credit_amount" in outliers
    assert outliers["credit_amount"]["skewness"] > 1.5
    assert outliers["credit_amount"]["outlier_count"] == 72
    assert outliers["duration_months"]["outlier_count"] == 70


def test_prepare_preprocessed_dataset_invariants() -> None:
    dataset = prepare_preprocessed_dataset(random_state=GLOBAL_RANDOM_SEED)

    # 1. Check partition dimensions
    assert len(dataset.X_train_raw) == 700
    assert len(dataset.X_test_raw) == 300
    assert dataset.X_train_processed.shape[0] == 700
    assert dataset.X_test_processed.shape[0] == 300
    assert dataset.X_train_processed.shape[1] == 48
    assert dataset.X_test_processed.shape[1] == 48

    # 2. Check no data leakage in scaling
    scaler_means = dataset.scaler.mean_
    train_num_means = dataset.X_train_raw[NUMERICAL_COLUMNS].mean().to_numpy()
    np.testing.assert_allclose(scaler_means, train_num_means, rtol=1e-5)

    # 3. Check deterministic reproducibility under seed 343
    dataset2 = prepare_preprocessed_dataset(random_state=GLOBAL_RANDOM_SEED)
    assert dataset.X_train_raw.index.equals(dataset2.X_train_raw.index)
    assert dataset.X_test_raw.index.equals(dataset2.X_test_raw.index)
    np.testing.assert_array_equal(dataset.X_train_processed, dataset2.X_train_processed)
