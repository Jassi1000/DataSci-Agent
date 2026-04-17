"""Tools module initialization.

Exports tool functions for convenient imports:
    from auto_ds_agent.tools import get_missing_summary, split_data, ...
"""

from auto_ds_agent.tools.data_tools import (
    get_missing_summary,
    impute_column,
    detect_outliers_iqr,
    fix_dtypes,
    encode_categoricals,
)
from auto_ds_agent.tools.ml_tools import (
    get_model_registry,
    split_data,
    scale_features,
    train_model,
    evaluate_classification,
    evaluate_regression,
    evaluate_clustering,
    cross_validate_model,
)
from auto_ds_agent.tools.viz_tools import (
    plot_missing_heatmap,
    plot_correlation_matrix,
    plot_distribution,
    plot_boxplots,
    plot_pairplot,
    generate_summary_stats,
)

__all__ = [
    # data_tools
    "get_missing_summary",
    "impute_column",
    "detect_outliers_iqr",
    "fix_dtypes",
    "encode_categoricals",
    # ml_tools
    "get_model_registry",
    "split_data",
    "scale_features",
    "train_model",
    "evaluate_classification",
    "evaluate_regression",
    "evaluate_clustering",
    "cross_validate_model",
    # viz_tools
    "plot_missing_heatmap",
    "plot_correlation_matrix",
    "plot_distribution",
    "plot_boxplots",
    "plot_pairplot",
    "generate_summary_stats",
]
