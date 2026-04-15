"""Test suite for DataSci-Agent — data_tools unit tests.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for test discovery
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import pytest

from tools.data_tools import (
    get_missing_summary,
    impute_column,
    detect_outliers_iqr,
    fix_dtypes,
    encode_categoricals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a small DataFrame with mixed types and missing values."""
    return pd.DataFrame({
        "age": [25, 30, np.nan, 45, 50, 35, np.nan, 28, 60, 22],
        "salary": [50000, 60000, 70000, np.nan, 90000, 55000, 65000, np.nan, 95000, 48000],
        "city": ["NY", "LA", "NY", "SF", "LA", np.nan, "NY", "SF", "LA", "NY"],
        "score": ["85", "90", "78", "92", "88", "76", "95", "80", "91", "87"],
        "joined": ["2020-01-15", "2019-06-20", "2021-03-10", "2018-11-05",
                    "2022-07-01", "2020-09-12", "2019-12-25", "2021-08-30",
                    "2017-04-18", "2023-01-01"],
        "constant": ["x", "x", "x", "x", "x", "x", "x", "x", "x", "x"],
    })


# ---------------------------------------------------------------------------
# Tests — get_missing_summary
# ---------------------------------------------------------------------------

class TestGetMissingSummary:
    def test_returns_all_columns(self, sample_df: pd.DataFrame) -> None:
        summary = get_missing_summary(sample_df)
        assert set(summary.keys()) == set(sample_df.columns)

    def test_correct_missing_count(self, sample_df: pd.DataFrame) -> None:
        summary = get_missing_summary(sample_df)
        assert summary["age"]["missing_count"] == 2
        assert summary["salary"]["missing_count"] == 2
        assert summary["city"]["missing_count"] == 1
        assert summary["score"]["missing_count"] == 0

    def test_missing_pct_range(self, sample_df: pd.DataFrame) -> None:
        summary = get_missing_summary(sample_df)
        for col_info in summary.values():
            assert 0.0 <= col_info["missing_pct"] <= 100.0


# ---------------------------------------------------------------------------
# Tests — impute_column
# ---------------------------------------------------------------------------

class TestImputeColumn:
    def test_mean_imputation(self, sample_df: pd.DataFrame) -> None:
        result = impute_column(sample_df, "age", "mean")
        assert result["age"].isna().sum() == 0

    def test_median_imputation(self, sample_df: pd.DataFrame) -> None:
        result = impute_column(sample_df, "salary", "median")
        assert result["salary"].isna().sum() == 0

    def test_mode_imputation(self, sample_df: pd.DataFrame) -> None:
        result = impute_column(sample_df, "city", "mode")
        assert result["city"].isna().sum() == 0

    def test_drop_strategy(self, sample_df: pd.DataFrame) -> None:
        result = impute_column(sample_df, "age", "drop")
        assert result["age"].isna().sum() == 0
        assert len(result) == 8  # 2 rows dropped

    def test_does_not_mutate_original(self, sample_df: pd.DataFrame) -> None:
        original_na = sample_df["age"].isna().sum()
        _ = impute_column(sample_df, "age", "mean")
        assert sample_df["age"].isna().sum() == original_na


# ---------------------------------------------------------------------------
# Tests — detect_outliers_iqr
# ---------------------------------------------------------------------------

class TestDetectOutliersIQR:
    def test_returns_dict_with_column_key(self, sample_df: pd.DataFrame) -> None:
        result = detect_outliers_iqr(sample_df, "age")
        assert "age" in result

    def test_non_numeric_returns_empty(self, sample_df: pd.DataFrame) -> None:
        result = detect_outliers_iqr(sample_df, "city")
        assert result["city"] == []

    def test_missing_col_returns_empty(self, sample_df: pd.DataFrame) -> None:
        result = detect_outliers_iqr(sample_df, "nonexistent")
        assert result["nonexistent"] == []


# ---------------------------------------------------------------------------
# Tests — fix_dtypes
# ---------------------------------------------------------------------------

class TestFixDtypes:
    def test_numeric_string_converted(self, sample_df: pd.DataFrame) -> None:
        result = fix_dtypes(sample_df)
        assert pd.api.types.is_numeric_dtype(result["score"])

    def test_date_string_converted(self, sample_df: pd.DataFrame) -> None:
        result = fix_dtypes(sample_df)
        assert pd.api.types.is_datetime64_any_dtype(result["joined"])

    def test_does_not_mutate_original(self, sample_df: pd.DataFrame) -> None:
        original_dtype = sample_df["score"].dtype
        _ = fix_dtypes(sample_df)
        assert sample_df["score"].dtype == original_dtype


# ---------------------------------------------------------------------------
# Tests — encode_categoricals
# ---------------------------------------------------------------------------

class TestEncodeCategoricals:
    def test_single_value_column_dropped(self, sample_df: pd.DataFrame) -> None:
        result = encode_categoricals(sample_df)
        assert "constant" not in result.columns

    def test_low_cardinality_one_hot(self, sample_df: pd.DataFrame) -> None:
        result = encode_categoricals(sample_df)
        # "city" has <=10 unique values → one-hot encoded → original col gone
        assert "city" not in result.columns
        # At least one dummy column should exist
        city_dummies = [c for c in result.columns if c.startswith("city_")]
        assert len(city_dummies) > 0

    def test_does_not_mutate_original(self, sample_df: pd.DataFrame) -> None:
        original_cols = sample_df.columns.tolist()
        _ = encode_categoricals(sample_df)
        assert sample_df.columns.tolist() == original_cols
