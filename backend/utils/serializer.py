# backend/utils/serializer.py
import json
import math

import numpy as np
import pandas as pd


def make_serializable(obj, _depth=0):
    """Recursively convert any object to JSON-safe Python types."""
    if _depth > 20:
        return str(obj)
    if isinstance(obj, dict):
        return {
            str(k): make_serializable(v, _depth + 1)
            for k, v in obj.items()
            if not _should_skip(k, v)
        }
    if isinstance(obj, (list, tuple, set)):
        return [make_serializable(i, _depth + 1) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist(), _depth + 1)
    if isinstance(obj, pd.DataFrame):
        return make_serializable(obj.head(20).to_dict(orient="records"), _depth + 1)
    if isinstance(obj, pd.Series):
        return make_serializable(obj.to_dict(), _depth + 1)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _should_skip(key, value):
    skip_keys = {
        "best_model_object",
        "model_object",
        "fitted_model",
        "X_train",
        "X_test",
        "y_train",
        "_estimator",
        "pipeline_object",
        "scaler_object",
    }
    if key in skip_keys:
        return True
    mod = getattr(type(value), "__module__", "") or ""
    return any(lib in mod for lib in ["sklearn", "xgboost", "lightgbm", "catboost"])
