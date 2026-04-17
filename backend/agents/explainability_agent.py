# backend/agents/explainability_agent.py
import joblib
import numpy as np

from utils.serializer import make_serializable

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ExplainabilityAgent:
    def run(self, model_path: str, X_train, X_test, feature_names: list, model_type: str, problem_type: str) -> dict:
        if not SHAP_AVAILABLE:
            return {"status": "skipped", "reason": "shap not installed", "shap_available": False}
        try:
            model = joblib.load(model_path)
            class_name = model.__class__.__name__
            X_train_sample = X_train[: min(len(X_train), 200)]
            X_test_sample = X_test[: min(len(X_test), 100)]
            if any(token in class_name for token in ["Forest", "XGB", "LGBM", "CatBoost", "Boosting", "Tree"]):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sample)
            elif any(token in class_name for token in ["LogisticRegression", "Ridge", "Lasso", "LinearRegression"]):
                explainer = shap.LinearExplainer(model, X_train_sample)
                shap_values = explainer.shap_values(X_test_sample)
            else:
                background = shap.sample(X_train_sample, min(20, len(X_train_sample)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_test_sample[:20])
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 0]
            mean_abs = np.abs(shap_values).mean(axis=0)
            importance = sorted([{"feature": str(feature_names[i]), "mean_abs_shap": float(v)} for i, v in enumerate(mean_abs[: len(feature_names)])], key=lambda x: x["mean_abs_shap"], reverse=True)
            top = importance[:5]
            if len(top) >= 3:
                plain = f"The top 3 most influential features were {top[0]['feature']}, {top[1]['feature']}, and {top[2]['feature']}. {top[0]['feature']} had a mean absolute SHAP value of {top[0]['mean_abs_shap']:.3f}, indicating it has the strongest impact on predictions."
            elif top:
                plain = f"The most influential feature was {top[0]['feature']}."
            else:
                plain = "SHAP ran successfully, but no feature importances were produced."
            return make_serializable({"status": "success", "global_importance": importance, "top_features": [x["feature"] for x in top], "plain_english": plain, "shap_available": True})
        except Exception as exc:
            return {"status": "error", "error": str(exc), "shap_available": True}

