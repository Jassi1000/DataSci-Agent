# backend/agents/ml_agent.py
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

from utils.serializer import make_serializable

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class MLAgent:
    def run(self, df: pd.DataFrame, target_col: str, problem_type: str, selected_models: list = None, user_params: dict = None, fast_mode: bool = False, progress_callback=None, run_id: str = "latest") -> dict:
        try:
            if target_col not in df.columns:
                return {"status": "error", "error": f"Target column '{target_col}' not found."}
            X = df.drop(columns=[target_col])
            y = df[target_col]
            if problem_type == "classification":
                y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)
            stratify = y if problem_type == "classification" and pd.Series(y).nunique() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

            X_fit, y_fit, imbalance_ratio, handled, smote = self._balance(X_train, y_train, problem_type)
            registry, unavailable = self._registry(problem_type, fast_mode)
            if not selected_models:
                selected_models = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier"] if problem_type == "classification" else ["LinearRegression", "RandomForestRegressor", "XGBRegressor"]
            if selected_models and not any(name in registry for name in selected_models):
                selected_models = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier"] if problem_type == "classification" else ["LinearRegression", "RandomForestRegressor", "XGBRegressor"]

            jobs, all_models = [], []
            for name in selected_models:
                if name in unavailable:
                    all_models.append(self._failed(name, unavailable[name]))
                elif name not in registry:
                    all_models.append(self._failed(name, "Model is not available in the registry."))
                else:
                    model = registry[name]
                    params = (user_params or {}).get(name, {})
                    if params:
                        model.set_params(**params)
                    jobs.append((name, model))

            try:
                trained = Parallel(n_jobs=-1, prefer="threads")(delayed(self._train_one)(name, model, X_fit, X_test, y_fit, y_test, problem_type) for name, model in jobs)
            except PermissionError:
                trained = [self._train_one(name, model, X_fit, X_test, y_fit, y_test, problem_type) for name, model in jobs]
            all_models.extend(trained)
            for result in trained:
                if progress_callback:
                    progress_callback(result.get("name", "model"), "done", f"Finished {result.get('name')}")

            metric = "f1_weighted" if problem_type == "classification" else "r2"
            ok = [m for m in all_models if not m.get("error") and m.get("metrics")]
            if not ok:
                return make_serializable({"status": "error", "error": "No models trained successfully.", "all_models": all_models})
            ok.sort(key=lambda m: m["metrics"].get(metric, -np.inf), reverse=True)
            for i, item in enumerate(ok, 1):
                item["rank"] = i

            self._tune(ok[:3], X_fit, X_test, y_fit, y_test, problem_type, fast_mode)
            ok.sort(key=lambda m: (m.get("tuned_metrics") or m.get("metrics")).get(metric, -np.inf), reverse=True)
            for i, item in enumerate(ok, 1):
                item["rank"] = i

            best = ok[0]
            model_dir = Path("./storage/models")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{run_id}.pkl"
            joblib.dump(best["fitted_model"], model_path)
            best_metrics = best.get("tuned_metrics") or best["metrics"]
            second = ok[1] if len(ok) > 1 else None
            second_score = ((second.get("tuned_metrics") or second.get("metrics")).get(metric, 0) if second else 0)
            delta = float(best_metrics.get(metric, 0) - second_score)
            params = best.get("tuned_params") or best["fitted_model"].get_params()
            why_best = f"{best['name']} achieved {metric}={best_metrics.get(metric, 0):.4f}, outperforming {second['name'] if second else 'the baseline'} by +{delta:.4f}. Best hyperparameters: {params}. Training time: {best.get('training_time_sec', 0):.1f}s."
            returned_models = [{k: v for k, v in m.items() if k not in {"fitted_model", "scaler_object", "y_pred"}} for m in all_models]
            y_pred = best.get("y_pred", [])
            result = {
                "status": "success",
                "problem_type": problem_type,
                "all_models": returned_models,
                "best_model_name": best["name"],
                "best_params": params,
                "best_metrics": best_metrics,
                "why_best": why_best,
                "feature_names": X.columns.tolist(),
                "preprocessing_notes": self._notes(problem_type, handled, smote),
                "class_imbalance_handled": handled,
                "smote_applied": smote,
                "imbalance_ratio": round(float(imbalance_ratio), 4),
                "train_size": int(len(X_fit)),
                "test_size": int(len(X_test)),
                "scaler_used": best.get("scaler_used", "none"),
                "model_path": str(model_path),
                "y_test": pd.Series(y_test).tolist(),
                "y_pred": pd.Series(y_pred).tolist(),
                "holdout_y_true": pd.Series(y_test).tolist(),
                "holdout_y_pred": pd.Series(y_pred).tolist(),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist() if problem_type == "classification" else [],
            }
            return make_serializable(result)
        except Exception as exc:
            return make_serializable({"status": "error", "error": str(exc)})

    def _balance(self, X_train, y_train, problem_type):
        imbalance_ratio, handled, smote = 1.0, False, False
        if problem_type != "classification":
            return X_train, y_train, imbalance_ratio, handled, smote
        counts = pd.Series(y_train).value_counts()
        if len(counts) > 1 and counts.min() > 0:
            imbalance_ratio = float(counts.max() / counts.min())
        if imbalance_ratio > 3:
            handled = True
            try:
                from imblearn.over_sampling import SMOTE
                X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
                smote = True
            except ImportError:
                smote = False
            except Exception:
                smote = False
        return X_train, y_train, imbalance_ratio, handled, smote

    def _registry(self, problem_type, fast_mode):
        unavailable = {}
        if problem_type == "classification":
            registry = {
                "LogisticRegression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced"),
                "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=None, random_state=42),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42),
                "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
                "SVC": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced"),
                "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5, weights="distance"),
            }
            if XGB_AVAILABLE:
                registry["XGBClassifier"] = XGBClassifier(n_estimators=300 if not fast_mode else 100, learning_rate=0.05, max_depth=6, subsample=0.8, eval_metric="logloss", verbosity=0, random_state=42)
            else:
                unavailable["XGBClassifier"] = "xgboost is not installed."
            if LGBM_AVAILABLE:
                registry["LGBMClassifier"] = LGBMClassifier(n_estimators=300 if not fast_mode else 100, learning_rate=0.05, num_leaves=31, class_weight="balanced", verbose=-1, random_state=42)
            else:
                unavailable["LGBMClassifier"] = "lightgbm is not installed."
            if CATBOOST_AVAILABLE:
                registry["CatBoostClassifier"] = CatBoostClassifier(iterations=300 if not fast_mode else 100, learning_rate=0.05, depth=6, verbose=0, random_state=42)
            else:
                unavailable["CatBoostClassifier"] = "catboost is not installed."
            return registry, unavailable
        registry = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
            "SVR": SVR(C=1.0, kernel="rbf"),
        }
        if XGB_AVAILABLE:
            registry["XGBRegressor"] = XGBRegressor(n_estimators=300 if not fast_mode else 100, learning_rate=0.05, max_depth=6, verbosity=0, random_state=42)
        else:
            unavailable["XGBRegressor"] = "xgboost is not installed."
        if LGBM_AVAILABLE:
            registry["LGBMRegressor"] = LGBMRegressor(n_estimators=300 if not fast_mode else 100, learning_rate=0.05, verbose=-1, random_state=42)
        else:
            unavailable["LGBMRegressor"] = "lightgbm is not installed."
        if CATBOOST_AVAILABLE:
            registry["CatBoostRegressor"] = CatBoostRegressor(iterations=300 if not fast_mode else 100, learning_rate=0.05, verbose=0, random_state=42)
        else:
            unavailable["CatBoostRegressor"] = "catboost is not installed."
        return registry, unavailable

    def _train_one(self, name, model, X_train, X_test, y_train, y_test, problem_type):
        started = time.time()
        try:
            scaler = None
            X_train_fit, X_test_eval = X_train, X_test
            if any(token in name for token in ["Logistic", "Linear", "Ridge", "Lasso", "SVC", "SVR", "KNeighbors"]):
                scaler = StandardScaler()
                X_train_fit = scaler.fit_transform(X_train)
                X_test_eval = scaler.transform(X_test)
            model.fit(X_train_fit, y_train)
            y_pred = model.predict(X_test_eval)
            return {"name": name, "display_name": name.replace("Classifier", "").replace("Regressor", ""), "metrics": self._metrics(model, X_test_eval, y_test, y_pred, problem_type), "training_time_sec": round(time.time() - started, 3), "tuned": False, "tuned_params": {}, "tuned_metrics": {}, "rank": None, "error": None, "fitted_model": model, "scaler_used": "StandardScaler" if scaler else "none", "scaler_object": scaler, "y_pred": y_pred}
        except Exception as exc:
            return self._failed(name, str(exc), time.time() - started)

    def _metrics(self, model, X_test, y_test, y_pred, problem_type):
        if problem_type == "classification":
            metrics = {"accuracy": accuracy_score(y_test, y_pred), "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0), "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0), "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0), "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(), "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)}
            if len(np.unique(y_test)) == 2:
                try:
                    scores = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                    metrics["roc_auc"] = roc_auc_score(y_test, scores)
                except Exception:
                    metrics["roc_auc"] = None
            return metrics
        mse = mean_squared_error(y_test, y_pred)
        denom = np.where(np.asarray(y_test) == 0, np.nan, np.asarray(y_test))
        mape = np.nanmean(np.abs((np.asarray(y_test) - np.asarray(y_pred)) / denom)) * 100
        return {"mae": mean_absolute_error(y_test, y_pred), "mse": mse, "rmse": float(np.sqrt(mse)), "r2": r2_score(y_test, y_pred), "mape": None if np.isnan(mape) else float(mape)}

    def _tune(self, top_models, X_train, X_test, y_train, y_test, problem_type, fast_mode):
        scoring = "f1_weighted" if problem_type == "classification" else "r2"
        for item in top_models:
            grid = self._grid(item["name"])
            if not grid:
                continue
            try:
                X_fit, X_eval = X_train, X_test
                if item.get("scaler_used") == "StandardScaler":
                    scaler = StandardScaler()
                    X_fit = scaler.fit_transform(X_train)
                    X_eval = scaler.transform(X_test)
                search = RandomizedSearchCV(item["fitted_model"], grid, n_iter=10 if fast_mode else 20, cv=3 if fast_mode else 5, scoring=scoring, n_jobs=1, random_state=42)
                search.fit(X_fit, y_train)
                pred = search.best_estimator_.predict(X_eval)
                item.update({"tuned": True, "tuned_params": search.best_params_, "tuned_metrics": self._metrics(search.best_estimator_, X_eval, y_test, pred, problem_type), "fitted_model": search.best_estimator_, "y_pred": pred})
            except Exception as exc:
                item["tuning_error"] = str(exc)

    def _grid(self, name):
        if "XGB" in name:
            return {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 6, 8], "subsample": [0.7, 0.8, 1.0]}
        if "LGBM" in name:
            return {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "num_leaves": [20, 31, 50]}
        if "RandomForest" in name:
            return {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]}
        if "CatBoost" in name:
            return {"iterations": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "depth": [4, 6, 8]}
        if "LogisticRegression" in name:
            return {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "saga"]}
        if "GradientBoosting" in name:
            return {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [2, 3, 4, 5]}
        return None

    def _failed(self, name, error, elapsed=0):
        return {"name": name, "display_name": name.replace("Classifier", "").replace("Regressor", ""), "metrics": {}, "training_time_sec": round(float(elapsed), 3), "tuned": False, "tuned_params": {}, "tuned_metrics": {}, "rank": None, "error": error}

    def _notes(self, problem_type, handled, smote):
        text = "StandardScaler was applied to linear, SVM, and KNN models. Tree models used cleaned numeric features without scaling."
        if problem_type == "classification" and handled:
            text += " Class imbalance was detected; SMOTE was applied." if smote else " Class imbalance was detected; class weights were used where supported."
        return text
