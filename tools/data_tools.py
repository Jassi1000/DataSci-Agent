"""Pure data manipulation and wrangling tools (Pandas/NumPy/scikit-learn).

These functions contain NO LLM logic and are designed to be called
by DataAgent or any other consumer that needs deterministic data ops.
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def get_missing_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a per-column missing-value report.

    Returns:
        dict keyed by column name, each value is a dict with:
            - missing_count: int
            - missing_pct: float (0-100)
            - dtype: str
            - unique_values: int (excluding NaN)
    """
    summary: Dict[str, Any] = {}
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        total = len(df)
        missing_pct = round((missing_count / total) * 100, 2) if total > 0 else 0.0
        unique_values = int(df[col].dropna().nunique())
        summary[col] = {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "dtype": str(df[col].dtype),
            "unique_values": unique_values,
        }
    logger.info("Missing summary computed for %d columns.", len(df.columns))
    return summary


def impute_column(df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
    """Impute a single column in-place copy using the given strategy.

    Args:
        df: Source DataFrame (not mutated; a copy is returned).
        col: Column name to impute.
        strategy: One of 'mean', 'median', 'mode', 'drop'.

    Returns:
        DataFrame with the column imputed.
    """
    df = df.copy()

    if col not in df.columns:
        logger.warning("Column '%s' not found in DataFrame. Skipping imputation.", col)
        return df

    if strategy == "drop":
        before = len(df)
        df = df.dropna(subset=[col])
        logger.info("Dropped %d rows with NaN in '%s'.", before - len(df), col)
        return df

    if strategy == "mean":
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].mean()
        else:
            logger.warning("Cannot compute mean for non-numeric column '%s'. Falling back to mode.", col)
            fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
    elif strategy == "median":
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
        else:
            logger.warning("Cannot compute median for non-numeric column '%s'. Falling back to mode.", col)
            fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
    elif strategy == "mode":
        mode_series = df[col].mode()
        fill_value = mode_series.iloc[0] if not mode_series.empty else None
    else:
        logger.error("Unknown imputation strategy '%s' for column '%s'. Skipping.", strategy, col)
        return df

    if fill_value is None:
        logger.warning("No valid fill value for column '%s'. Column may be entirely null.", col)
        return df

    df[col] = df[col].fillna(fill_value)
    logger.info("Imputed column '%s' using strategy '%s' (fill_value=%s).", col, strategy, fill_value)
    return df


def detect_outliers_iqr(df: pd.DataFrame, col: str) -> Dict[str, List[int]]:
    """Detect outlier row indices for a numeric column using IQR method.

    Args:
        df: Source DataFrame.
        col: Numeric column name.

    Returns:
        Dict with the column name mapped to a list of integer row indices
        that are outliers.  Returns empty list for non-numeric columns.
    """
    if col not in df.columns:
        logger.warning("Column '%s' not found. Returning empty outlier set.", col)
        return {col: []}

    if not pd.api.types.is_numeric_dtype(df[col]):
        return {col: []}

    series = df[col].dropna()
    if series.empty:
        return {col: []}

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        # All non-null values identical – no meaningful outlier detection
        return {col: []}

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    outlier_indices = df.index[outlier_mask].tolist()

    logger.info(
        "Column '%s': IQR=%.4f, bounds=[%.4f, %.4f], outliers=%d",
        col, iqr, lower_bound, upper_bound, len(outlier_indices),
    )
    return {col: outlier_indices}


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to cast object columns to more appropriate dtypes.

    Heuristics applied (in order):
        1. Try pd.to_datetime – accept if ≥80 % of non-null values parse.
        2. Try pd.to_numeric  – accept if ≥80 % of non-null values parse.
        3. Otherwise keep as object.

    Returns:
        DataFrame with corrected dtypes (copy).
    """
    df = df.copy()
    PARSE_THRESHOLD = 0.80

    for col in df.select_dtypes(include=["object"]).columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue

        total = len(non_null)

        # --- attempt datetime ---
        try:
            parsed_dt = pd.to_datetime(non_null, infer_datetime_format=True, errors="coerce")
            success_rate = parsed_dt.notna().sum() / total
            if success_rate >= PARSE_THRESHOLD:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                logger.info("Converted column '%s' to datetime (success_rate=%.2f).", col, success_rate)
                continue
        except Exception:
            pass

        # --- attempt numeric ---
        try:
            parsed_num = pd.to_numeric(non_null, errors="coerce")
            success_rate = parsed_num.notna().sum() / total
            if success_rate >= PARSE_THRESHOLD:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                logger.info("Converted column '%s' to numeric (success_rate=%.2f).", col, success_rate)
                continue
        except Exception:
            pass

    return df


def encode_categoricals(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    """Encode categorical (object/category) columns.

    Strategy:
        - cardinality ≤ threshold  → one-hot encoding (pd.get_dummies)
        - cardinality >  threshold → label encoding (sklearn LabelEncoder)
        - single-value columns    → dropped (zero information)

    Args:
        df: Source DataFrame (not mutated).
        threshold: Max unique values for one-hot encoding.

    Returns:
        DataFrame with encoded categoricals.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    one_hot_cols: List[str] = []
    label_encode_cols: List[str] = []
    drop_cols: List[str] = []

    for col in cat_cols:
        nunique = df[col].dropna().nunique()

        if nunique <= 1:
            drop_cols.append(col)
            logger.info("Dropping single-value / empty categorical column '%s'.", col)
            continue

        if nunique <= threshold:
            one_hot_cols.append(col)
        else:
            label_encode_cols.append(col)

    # Drop zero-information columns
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # One-hot encode low-cardinality columns
    if one_hot_cols:
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype=int)
        logger.info("One-hot encoded columns: %s", one_hot_cols)

    # Label encode high-cardinality columns
    for col in label_encode_cols:
        le = LabelEncoder()
        non_null_mask = df[col].notna()
        df.loc[non_null_mask, col] = le.fit_transform(df.loc[non_null_mask, col].astype(str))
        df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.info("Label-encoded column '%s' (%d unique values).", col, df[col].nunique())

    return df
