"""Data visualization generation tools.

Pure functions for creating charts and statistical plots.
No LLM logic — consumed by EDAAgent and ReporterAgent.
"""

import logging
import os
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# We use matplotlib with the Agg backend so it works headless (Docker, CI).
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_missing_heatmap(df: pd.DataFrame, output_path: str) -> str:
    """Generate a heatmap showing missing value patterns.

    Args:
        df: Source DataFrame.
        output_path: File path to save the PNG.

    Returns:
        Absolute path to the saved image.
    """
    fig, ax = plt.subplots(figsize=(12, max(4, len(df.columns) * 0.35)))
    sns.heatmap(df.isna().astype(int), cbar=True, yticklabels=False, cmap="YlOrRd", ax=ax)
    ax.set_title("Missing Value Heatmap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Missing-value heatmap saved to %s", output_path)
    return os.path.abspath(output_path)


def plot_correlation_matrix(df: pd.DataFrame, output_path: str) -> str:
    """Generate a correlation matrix heatmap for numeric columns.

    Args:
        df: Source DataFrame.
        output_path: File path to save the PNG.

    Returns:
        Absolute path to the saved image.
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        logger.warning("No numeric columns found. Skipping correlation matrix.")
        return ""

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr.columns) * 0.7), max(6, len(corr.columns) * 0.6)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Correlation matrix saved to %s", output_path)
    return os.path.abspath(output_path)


def plot_distribution(series: pd.Series, col_name: str, output_path: str) -> str:
    """Plot histogram + KDE for a numeric column, or countplot for categorical.

    Args:
        series: Column data.
        col_name: Column name (used in title).
        output_path: File path to save the PNG.

    Returns:
        Absolute path to the saved image.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if pd.api.types.is_numeric_dtype(series):
        series.dropna().hist(bins=30, ax=ax, edgecolor="black", alpha=0.7)
        ax.set_title(f"Distribution of {col_name}")
        ax.set_xlabel(col_name)
        ax.set_ylabel("Frequency")
    else:
        value_counts = series.value_counts().head(20)
        value_counts.plot.bar(ax=ax, edgecolor="black", alpha=0.7)
        ax.set_title(f"Top Categories in {col_name}")
        ax.set_xlabel(col_name)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Distribution plot for '%s' saved to %s", col_name, output_path)
    return os.path.abspath(output_path)


def plot_boxplots(df: pd.DataFrame, output_path: str, max_cols: int = 15) -> str:
    """Generate side-by-side boxplots for numeric columns.

    Args:
        df: Source DataFrame.
        output_path: File path to save the PNG.
        max_cols: Maximum number of columns to plot (avoids cluttered charts).

    Returns:
        Absolute path to the saved image.
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        logger.warning("No numeric columns found. Skipping boxplots.")
        return ""

    cols = numeric_df.columns[:max_cols].tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.2), 6))
    numeric_df[cols].boxplot(ax=ax, vert=True, patch_artist=True)
    ax.set_title("Boxplots of Numeric Features")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Boxplots saved to %s", output_path)
    return os.path.abspath(output_path)


def plot_pairplot(df: pd.DataFrame, output_path: str, max_cols: int = 6) -> str:
    """Generate a seaborn pairplot for a subset of numeric columns.

    Args:
        df: Source DataFrame.
        output_path: File path to save the PNG.
        max_cols: Maximum columns to include (pairplots grow O(n²)).

    Returns:
        Absolute path to the saved image.
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        logger.warning("Fewer than 2 numeric columns. Skipping pairplot.")
        return ""

    subset = numeric_df.iloc[:, :max_cols]
    g = sns.pairplot(subset, diag_kind="kde", plot_kws={"alpha": 0.5, "s": 15})
    g.fig.suptitle("Pairplot of Numeric Features", y=1.02)
    g.savefig(output_path, dpi=100)
    plt.close(g.fig)
    logger.info("Pairplot saved to %s", output_path)
    return os.path.abspath(output_path)


def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute descriptive statistics for the DataFrame.

    Returns:
        Dictionary with 'numeric' and 'categorical' summary tables
        serialised as dicts.
    """
    stats: Dict[str, Any] = {}

    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        stats["numeric"] = numeric_df.describe().to_dict()

    cat_df = df.select_dtypes(include=["object", "category"])
    if not cat_df.empty:
        stats["categorical"] = cat_df.describe().to_dict()

    stats["shape"] = list(df.shape)
    stats["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    logger.info("Summary statistics computed.")
    return stats
