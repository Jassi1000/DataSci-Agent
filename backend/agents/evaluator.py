# backend/agents/evaluator.py
import numpy as np
from sklearn.metrics import auc, classification_report, precision_recall_curve, roc_curve

from utils.serializer import make_serializable


class EvaluatorAgent:
    def run(self, ml_results: dict, y_test, y_pred, problem_type: str) -> dict:
        try:
            y_true, y_hat, warnings = self._align_inputs(y_test, y_pred, ml_results)
            if len(y_true) == 0 or len(y_hat) == 0:
                return make_serializable(
                    {
                        "status": "partial",
                        "problem_type": problem_type,
                        "warnings": warnings
                        + [
                            "Evaluation labels or predictions were unavailable. Showing model metrics captured during training instead."
                        ],
                        "recommendation": self._training_recommendation(ml_results, problem_type),
                        "training_metrics": ml_results.get("best_metrics", {}),
                        "per_class_metrics": ml_results.get("best_metrics", {}).get("classification_report", {}),
                        "roc_curve": {},
                        "precision_recall_curve": {},
                        "residual_analysis": [],
                        "error_distribution": {"counts": [], "bins": []},
                    }
                )

            if problem_type == "classification":
                roc_data = {}
                pr_data = {}
                if len(np.unique(y_true)) == 2:
                    try:
                        fpr, tpr, _ = roc_curve(y_true, y_hat)
                        precision, recall, _ = precision_recall_curve(y_true, y_hat)
                        name = ml_results.get("best_model_name", "best_model")
                        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
                        pr_data[name] = {"precision": precision.tolist(), "recall": recall.tolist()}
                    except Exception as exc:
                        warnings.append(f"ROC/PR curve generation skipped: {exc}")
                per_class = classification_report(y_true, y_hat, output_dict=True, zero_division=0)
                score = ml_results.get("best_metrics", {}).get("f1_weighted")
                recommendation = f"{ml_results.get('best_model_name', 'The selected model')} is recommended because it delivered a weighted F1 score of {score:.4f} on the holdout set." if isinstance(score, (int, float)) else "The selected model is recommended based on the strongest holdout performance."
                return make_serializable({"status": "success", "problem_type": "classification", "roc_curve": roc_data, "precision_recall_curve": pr_data, "per_class_metrics": per_class, "warnings": warnings, "recommendation": recommendation})

            actual = np.asarray(y_true, dtype=float)
            predicted = np.asarray(y_hat, dtype=float)
            residuals = actual - predicted
            counts, bins = np.histogram(residuals, bins=20)
            points = [{"actual": float(a), "predicted": float(p), "residual": float(r)} for a, p, r in zip(actual[:100], predicted[:100], residuals[:100])]
            recommendation = f"{ml_results.get('best_model_name', 'The selected model')} is recommended with R2={ml_results.get('best_metrics', {}).get('r2', 0):.4f} on the holdout set."
            return make_serializable({"status": "success", "problem_type": "regression", "residual_analysis": points, "error_distribution": {"counts": counts.tolist(), "bins": bins.tolist()}, "warnings": warnings, "recommendation": recommendation})
        except Exception as exc:
            return make_serializable({"status": "partial", "problem_type": problem_type, "error": str(exc), "warnings": [str(exc)], "recommendation": self._training_recommendation(ml_results, problem_type), "training_metrics": ml_results.get("best_metrics", {})})

    def _align_inputs(self, y_test, y_pred, ml_results):
        warnings = []
        y_true = list(y_test or [])
        y_hat = list(y_pred or [])
        if not y_true:
            y_true = list(ml_results.get("holdout_y_true") or [])
            if y_true:
                warnings.append("Recovered evaluation labels from holdout_y_true.")
        if not y_hat:
            y_hat = list(ml_results.get("holdout_y_pred") or [])
            if y_hat:
                warnings.append("Recovered predictions from holdout_y_pred.")
        if len(y_true) != len(y_hat):
            warnings.append(f"Evaluation input length mismatch detected: y_true={len(y_true)}, y_pred={len(y_hat)}.")
            if y_true and y_hat:
                n = min(len(y_true), len(y_hat))
                y_true = y_true[:n]
                y_hat = y_hat[:n]
                warnings.append(f"Aligned evaluation arrays by truncating to {n} paired samples.")
        return y_true, y_hat, warnings

    def _training_recommendation(self, ml_results, problem_type):
        metrics = ml_results.get("best_metrics", {})
        metric_name = "f1_weighted" if problem_type == "classification" else "r2"
        score = metrics.get(metric_name)
        if isinstance(score, (int, float)):
            return f"{ml_results.get('best_model_name', 'The selected model')} remains the recommended model based on training-stage holdout {metric_name}={score:.4f}."
        return f"{ml_results.get('best_model_name', 'The selected model')} remains the recommended model based on the available training-stage metrics."
