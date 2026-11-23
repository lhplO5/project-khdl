"""Bộ kiểm thử tự động cho pipeline dữ liệu Tuần 1."""

from src.clean import clean_dataset
from src.profiling import audit_dataframe
from src.run_sprint1 import generate_synthetic_carbonglobe


def test_synthetic_schema_and_audit() -> None:
    df = generate_synthetic_carbonglobe(n_sites=10, n_years=40, n_features=136, n_targets=7)
    assert df.shape == (400, 145)  # site_id + year + 136 features + 7 targets

    report = audit_dataframe(df, key_cols=["site_id", "year"])
    assert report["n_rows"] == 400
    assert report["duplicate_keys"] == 0
    assert report["n_cols_with_missing"] > 0


def test_clean_dataset_removes_nulls_and_preserves_keys() -> None:
    df = generate_synthetic_carbonglobe(n_sites=10, n_years=40, n_features=20, n_targets=3)
    cleaned, summary = clean_dataset(
        df,
        group_col="site_id",
        time_col="year",
        protected_cols=["site_id", "year"],
    )
    assert summary["remaining_nulls"] == 0
    assert "site_id" in cleaned.columns and "year" in cleaned.columns
    assert cleaned["feat_000"].max() < 999.0
