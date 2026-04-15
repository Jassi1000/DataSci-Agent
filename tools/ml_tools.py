"""Machine learning and evaluation tools (scikit-learn).

Pure functions — no LLM logic. Consumed by MLAgent and EvaluatorAgent.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — maps human-readable names to sklearn constructors
# ---------------------------------------------------------------------------

CLASSIFICATION_MODELS: Dict[str, Any] = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest_classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting_classifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "decision_tree_classifier": DecisionTreeClassifier(random_state=42),
    "svc": SVC(random_state=42),
}

REGRESSION_MODELS: Dict[str, Any] = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "lasso": Lasso(alpha=1.0),
    "random_forest_regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting_regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "decision_tree_regressor": DecisionTreeRegressor(random_state=42),
    "svr": SVR(),
}

CLUSTERING_MODELS: Dict[str, Any] = {
    "kmeans": KMeans(n_clusters=3, random_state=42, n_init=10),
    "dbscan": DBSCAN(),
}


def get_model_registry(problem_type: str) -> Dict[str, Any]:
    """Return the appropriate model registry for the problem type.

    Args:
        problem_type: One of 'classification', 'regression', 'clustering'.

    Returns:
        Dict mapping model name → unfitted estimator instance.
    """
    registries = {
        "classification": CLASSIFICATION_MODELS,
        "regression": REGRESSION_MODELS,
        "clustering": CLUSTERING_MODELS,
    }
    registry = registries.get(problem_type)
    if registry is None:
        logger.error("Unknown problem_type '%s'. Returning empty registry.", problem_type)
        return {}
    return registry


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split DataFrame into train/test sets.

    Args:
        df: Cleaned DataFrame.
        target_col: Name of the target column.
        test_size: Proportion held out for testing.
        random_state: Reproducibility seed.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Data split complete. Train: %d rows, Test: %d rows.",
        len(X_train), len(X_test),
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Apply StandardScaler fit on train, transform both sets.

    Returns:
        (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    logger.info("Feature scaling applied.")
    return X_train_scaled, X_test_scaled, scaler


def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
) -> Any:
    """Fit a scikit-learn estimator.

    Args:
        model: Unfitted estimator.
        X_train: Training features.
        y_train: Training target. Pass None for unsupervised models (clustering).

    Returns:
        Fitted estimator.
    """
    import sklearn.base as skbase
    model_clone = skbase.clone(model)
    if y_train is not None and len(y_train) > 0:
        model_clone.fit(X_train, y_train)
    else:
        model_clone.fit(X_train)
    logger.info("Model '%s' trained successfully.", type(model_clone).__name__)
    return model_clone


def evaluate_classification(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate a fitted classification model.

    Returns:
        Dict with accuracy, precision, recall, f1, confusion_matrix,
        and full classification_report.
    """
    y_pred = model.predict(X_test)
    avg = "weighted" if len(np.unique(y_test)) > 2 else "binary"

    results: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }
    logger.info(
        "Classification evaluation — accuracy: %.4f, f1: %.4f",
        results["accuracy"], results["f1_score"],
    )
    return results


def evaluate_regression(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate a fitted regression model.

    Returns:
        Dict with r2, mae, mse, rmse.
    """
    y_pred = model.predict(X_test)
    mse_val = float(mean_squared_error(y_test, y_pred))
    results: Dict[str, Any] = {
        "r2_score": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": mse_val,
        "rmse": float(np.sqrt(mse_val)),
    }
    logger.info("Regression evaluation — R²: %.4f, RMSE: %.4f", results["r2_score"], results["rmse"])
    return results


def evaluate_clustering(
    model: Any,
    X: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate a fitted clustering model.

    Returns:
        Dict with n_clusters, silhouette_score (if applicable),
        and cluster_sizes.
    """
    labels = model.labels_
    n_clusters = len(set(labels) - {-1})

    results: Dict[str, Any] = {
        "n_clusters": n_clusters,
        "cluster_sizes": pd.Series(labels).value_counts().to_dict(),
    }

    if n_clusters >= 2 and len(set(labels) - {-1}) < len(X):
        try:
            sil = float(silhouette_score(X, labels))
            results["silhouette_score"] = sil
            logger.info("Clustering evaluation — silhouette: %.4f, clusters: %d", sil, n_clusters)
        except Exception as exc:
            logger.warning("Silhouette score failed: %s", exc)
            results["silhouette_score"] = None
    else:
        results["silhouette_score"] = None

    return results


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "accuracy",
) -> Dict[str, Any]:
    """Run k-fold cross-validation.

    Args:
        model: Unfitted estimator (will be cloned internally by sklearn).
        X: Feature matrix.
        y: Target vector.
        cv: Number of folds.
        scoring: Metric name compatible with sklearn.

    Returns:
        Dict with mean, std, and per-fold scores.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    result = {
        "scoring": scoring,
        "cv_folds": cv,
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "fold_scores": scores.tolist(),
    }
    logger.info(
        "Cross-validation (%s, %d folds) — mean: %.4f ± %.4f",
        scoring, cv, result["mean_score"], result["std_score"],
    )
    return result
