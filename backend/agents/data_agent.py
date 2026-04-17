# backend/agents/data_agent.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataCleaningAgent:
    def run(self, df: pd.DataFrame, target_col: str = None) -> dict:
        try:
            cleaned = df.copy()
            original_shape = cleaned.shape
            original_missing = int(cleaned.isna().sum().sum())
            original_duplicates = int(cleaned.duplicated().sum())
            original_cells = max(int(cleaned.shape[0] * cleaned.shape[1]), 1)
            cleaning_log = []
            encoders = {}
            original_types = {col: str(dtype) for col, dtype in cleaned.dtypes.items()}

            missing_pct_by_col = cleaned.isna().mean()
            drop_cols = missing_pct_by_col[missing_pct_by_col > 0.7].index.tolist()
            if drop_cols:
                cleaned = cleaned.drop(columns=drop_cols)
                cleaning_log.append(f"Dropped {len(drop_cols)} columns with more than 70% missing values: {', '.join(drop_cols)}")

            for col in cleaned.columns:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    missing = int(cleaned[col].isna().sum())
                    if missing:
                        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                        cleaning_log.append(f"Filled {missing} missing numeric values in '{col}' with median.")
                else:
                    missing = int(cleaned[col].isna().sum())
                    if missing:
                        mode = cleaned[col].mode(dropna=True)
                        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
                        cleaned[col] = cleaned[col].fillna(fill_value)
                        cleaning_log.append(f"Filled {missing} missing categorical values in '{col}' with mode.")

            before_dupes = len(cleaned)
            cleaned = cleaned.drop_duplicates()
            duplicates_removed = before_dupes - len(cleaned)
            if duplicates_removed:
                cleaning_log.append(f"Removed {duplicates_removed} exact duplicate rows.")

            for col in cleaned.select_dtypes(include=["object", "string"]).columns:
                cleaned[col] = cleaned[col].astype(str).str.strip()
                cleaning_log.append(f"Stripped leading and trailing whitespace from '{col}'.")

            for col in cleaned.select_dtypes(include=["object", "string", "category", "bool"]).columns:
                encoder = LabelEncoder()
                cleaned[col] = encoder.fit_transform(cleaned[col].astype(str))
                encoders[col] = encoder
                cleaning_log.append(f"Label encoded categorical column '{col}' with {len(encoder.classes_)} classes.")

            zero_variance_cols = [
                col for col in cleaned.columns if cleaned[col].nunique(dropna=False) <= 1 and col != target_col
            ]
            if zero_variance_cols:
                cleaned = cleaned.drop(columns=zero_variance_cols)
                cleaning_log.append(f"Dropped {len(zero_variance_cols)} zero-variance columns: {', '.join(zero_variance_cols)}")

            missing_fixed = max(original_missing - int(cleaned.isna().sum().sum()), 0)
            rows_removed = original_shape[0] - cleaned.shape[0]
            cols_removed = original_shape[1] - cleaned.shape[1]
            missing_pct = original_missing / original_cells * 100
            duplicate_pct = original_duplicates / max(original_shape[0], 1) * 100
            cols_removed_pct = cols_removed / max(original_shape[1], 1) * 100
            data_quality_score = float(np.clip(100 - (missing_pct * 0.5 + duplicate_pct * 0.3 + cols_removed_pct * 0.2), 0, 100))

            if not cleaning_log:
                cleaning_log.append("Dataset already met cleaning requirements; no structural changes were needed.")

            return {
                "status": "success",
                "cleaned_df": cleaned,
                "original_shape": tuple(original_shape),
                "cleaned_shape": tuple(cleaned.shape),
                "rows_removed": int(rows_removed),
                "cols_removed": int(cols_removed),
                "missing_fixed": int(missing_fixed),
                "duplicates_removed": int(duplicates_removed),
                "data_quality_score": round(data_quality_score, 2),
                "cleaning_log": cleaning_log,
                "column_types": {
                    col: {"original": original_types.get(col, "unknown"), "cleaned": str(dtype)}
                    for col, dtype in cleaned.dtypes.items()
                },
                "encoders": encoders,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

