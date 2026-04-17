# backend/agents/eda_agent.py
import base64
from io import BytesIO

import numpy as np
import pandas as pd


class EDAAgent:
    def run(self, df: pd.DataFrame, target_col: str = None, problem_type: str = None) -> dict:
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            missing_per_column = {col: int(val) for col, val in df.isna().sum().items()}
            recommended_target = target_col or self._recommend_target(df)
            class_balance = {}

            if problem_type == "classification" and recommended_target in df.columns:
                counts = df[recommended_target].value_counts(dropna=False)
                total = max(int(counts.sum()), 1)
                class_balance = {
                    str(label): {"count": int(count), "percentage": round(float(count) / total * 100, 2)}
                    for label, count in counts.items()
                }

            feature_distributions = {}
            for col in numeric_df.columns:
                values = numeric_df[col].dropna()
                if values.empty:
                    continue
                counts, bins = np.histogram(values, bins=20)
                feature_distributions[col] = {
                    "counts": counts.tolist(),
                    "bins": bins.tolist(),
                    "skewness": float(values.skew()) if len(values) > 2 else 0.0,
                }

            insights = self._build_insights(df, numeric_df, class_balance)
            charts = {
                "correlation_matrix": self._chart_correlation(numeric_df),
                "boxplots": self._chart_boxplots(numeric_df),
                "missing_heatmap": self._chart_missing(df),
                "distributions": {
                    col: self._chart_distribution(numeric_df[col], col)
                    for col in numeric_df.columns[:5]
                },
            }

            return {
                "status": "success",
                "problem_type": problem_type,
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_per_column": missing_per_column,
                "describe": df.describe(include="all").fillna("").to_dict(),
                "correlation": numeric_df.corr(numeric_only=True).fillna(0).to_dict(),
                "class_balance": class_balance,
                "feature_distributions": feature_distributions,
                "recommended_target": recommended_target,
                "insights": insights,
                "charts": charts,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def _recommend_target(self, df):
        if df.empty:
            return None
        scores = {}
        rows = max(len(df), 1)
        for col in df.columns:
            unique_ratio = df[col].nunique(dropna=False) / rows
            if pd.api.types.is_numeric_dtype(df[col]):
                scores[col] = float(df[col].var() or 0) + unique_ratio
            else:
                scores[col] = unique_ratio * 10
        return max(scores, key=scores.get)

    def _build_insights(self, df, numeric_df, class_balance):
        insights = []
        for col, pct in df.isna().mean().items():
            if pct > 0.3:
                insights.append(
                    {
                        "severity": "HIGH",
                        "topic": "High missingness",
                        "description": f"Column '{col}' has {pct * 100:.1f}% missing values.",
                    }
                )
        if class_balance:
            counts = [v["count"] for v in class_balance.values()]
            if counts and min(counts) > 0 and max(counts) / min(counts) > 5:
                insights.append(
                    {
                        "severity": "HIGH",
                        "topic": "Class imbalance",
                        "description": f"The largest class is {max(counts) / min(counts):.1f}x the smallest class.",
                    }
                )
        corr = numeric_df.corr(numeric_only=True).abs()
        for i, col in enumerate(corr.columns):
            for other in corr.columns[i + 1 :]:
                value = corr.loc[col, other]
                if pd.notna(value) and value > 0.9:
                    insights.append(
                        {
                            "severity": "MEDIUM",
                            "topic": "High correlation",
                            "description": f"'{col}' and '{other}' are strongly correlated ({value:.2f}).",
                        }
                    )
        for col in numeric_df.columns:
            if numeric_df[col].nunique(dropna=False) <= 2:
                insights.append(
                    {
                        "severity": "MEDIUM",
                        "topic": "Low variance",
                        "description": f"'{col}' has very few distinct values and may contribute little signal.",
                    }
                )
            skewness = numeric_df[col].dropna().skew() if len(numeric_df[col].dropna()) > 2 else 0
            if pd.notna(skewness) and abs(skewness) > 2:
                insights.append(
                    {
                        "severity": "LOW",
                        "topic": "Skewed distribution",
                        "description": f"'{col}' is highly skewed (skewness {skewness:.2f}).",
                    }
                )
        if not insights:
            insights.append(
                {
                    "severity": "LOW",
                    "topic": "Stable profile",
                    "description": "No severe data quality or distribution risks were detected in the cleaned dataset.",
                }
            )
        return insights[:20]

    def _encode_plot(self, plot_fn):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            plot_fn(plt, sns)
            buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
            plt.close()
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception:
            return {"type": "unavailable", "reason": "Chart generation requires kaleido or a plotting backend."}

    def _chart_correlation(self, numeric_df):
        if numeric_df.shape[1] < 2:
            return None
        return self._encode_plot(
            lambda plt, sns: (
                plt.figure(figsize=(8, 6)),
                sns.heatmap(numeric_df.corr(numeric_only=True), cmap="viridis", annot=False),
                plt.title("Correlation Matrix"),
            )
        )

    def _chart_boxplots(self, numeric_df):
        if numeric_df.empty:
            return None
        cols = numeric_df.columns[:10]
        return self._encode_plot(
            lambda plt, sns: (
                plt.figure(figsize=(10, 5)),
                sns.boxplot(data=numeric_df[cols], orient="h"),
                plt.title("Numeric Feature Boxplots"),
            )
        )

    def _chart_missing(self, df):
        missing = df.isna().sum()
        if int(missing.sum()) == 0:
            missing = pd.Series({col: 0 for col in df.columns[:10]})
        return self._encode_plot(
            lambda plt, sns: (
                plt.figure(figsize=(9, 4)),
                sns.barplot(x=missing.index[:20], y=missing.values[:20]),
                plt.xticks(rotation=45, ha="right"),
                plt.title("Missing Values by Column"),
            )
        )

    def _chart_distribution(self, series, col):
        values = series.dropna()
        if values.empty:
            return None
        return self._encode_plot(
            lambda plt, sns: (
                plt.figure(figsize=(7, 4)),
                sns.histplot(values, kde=True, bins=20),
                plt.title(f"Distribution: {col}"),
            )
        )

