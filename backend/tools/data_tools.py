# backend/tools/data_tools.py
import pandas as pd


def dataframe_info(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "unique_counts": {col: int(df[col].nunique(dropna=False)) for col in df.columns},
        "missing_counts": {col: int(df[col].isna().sum()) for col in df.columns},
    }

