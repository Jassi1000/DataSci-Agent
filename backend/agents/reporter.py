# backend/agents/reporter.py


class ReporterAgent:
    def run(self, plan: dict, cleaning: dict, eda: dict, ml: dict, explainability: dict, evaluation: dict, dataset_name: str) -> dict:
        try:
            best_model = ml.get("best_model_name", "the selected model")
            problem_type = ml.get("problem_type", "machine learning")
            metrics = ml.get("best_metrics", {})
            primary_value = metrics.get("f1_weighted") if problem_type == "classification" else metrics.get("r2")
            quality = cleaning.get("data_quality_score", 0)
            shape = eda.get("shape", ["unknown", "unknown"])
            score_text = f"The best score was {primary_value:.4f}. " if isinstance(primary_value, (int, float)) else ""
            executive_summary = f"InsightForge analyzed {dataset_name} with {shape[0]} rows and {shape[1]} columns. The workflow identified a {problem_type} task and selected {best_model} as the strongest model. {score_text}Data quality scored {quality}/100 after automated cleaning."
            key_findings = [
                f"Data cleaning removed {cleaning.get('rows_removed', 0)} rows and {cleaning.get('cols_removed', 0)} columns.",
                f"Missing value handling fixed {cleaning.get('missing_fixed', 0)} cells.",
                f"The best model was {best_model}.",
                ml.get("why_best", "Model comparison completed successfully."),
                explainability.get("plain_english", "Explainability was unavailable for this run."),
            ]
            if eda.get("insights"):
                key_findings.append(eda["insights"][0].get("description", "EDA completed."))
            recommendations = [
                "Review the top-ranked features with domain experts before operational use.",
                "Validate the selected model on fresh, unseen data before production deployment.",
                "Monitor data drift and retrain when feature distributions or target balance change materially.",
            ]
            if ml.get("class_imbalance_handled"):
                recommendations[0] = "Track class-level performance because imbalance was detected during training."
            verdict = "Strong data quality after cleaning." if quality >= 80 else "Moderate data quality; review cleaning decisions before relying on downstream metrics." if quality >= 60 else "Low data quality; collect or repair source data before using model outputs for important decisions."
            next_steps = ["Export the report for stakeholder review.", "Use chat to inspect model behavior and key findings.", "Schedule a holdout validation pass with new data."]
            top_features = explainability.get("top_features") or []
            if top_features:
                next_steps.append(f"Investigate business meaning for top feature '{top_features[0]}'.")
            return {"status": "success", "executive_summary": executive_summary, "key_findings": key_findings[:5], "recommendations": recommendations[:3], "model_conclusion": evaluation.get("recommendation") or f"{best_model} is the recommended model for this dataset.", "data_quality_verdict": verdict, "next_steps": next_steps}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

