# backend/agents/planner.py
import pandas as pd


class PlannerAgent:
    def run(self, goal: str, df_info: dict, target_col: str = None) -> dict:
        try:
            columns = df_info.get("columns", [])
            dtypes = df_info.get("dtypes", {})
            detected_target = target_col or (columns[-1] if columns else None)
            dtype = str(dtypes.get(detected_target, "")).lower()
            unique_counts = df_info.get("unique_counts", {})
            row_count = max(int(df_info.get("rows", 0) or 0), 1)
            unique_ratio = (unique_counts.get(detected_target, row_count) or row_count) / row_count

            if detected_target and (
                "object" in dtype
                or "category" in dtype
                or "bool" in dtype
                or unique_ratio < 0.2
                or unique_counts.get(detected_target, row_count) <= 20
            ):
                problem_type = "classification"
            else:
                problem_type = "regression"

            plan = [
                {
                    "step": 1,
                    "agent": "data_cleaning",
                    "action": "Clean missing values, duplicates, categorical encodings, and zero-variance features.",
                    "reasoning": "Reliable modeling requires consistent numeric inputs and a clear audit trail.",
                },
                {
                    "step": 2,
                    "agent": "eda",
                    "action": "Profile distributions, missingness, correlations, target balance, and notable risks.",
                    "reasoning": "Exploratory analysis exposes quality issues and informs model selection.",
                },
                {
                    "step": 3,
                    "agent": "ml_training",
                    "action": f"Train and compare candidate {problem_type} models.",
                    "reasoning": "Multiple model families reduce the chance of overfitting to one algorithmic assumption.",
                },
                {
                    "step": 4,
                    "agent": "explainability",
                    "action": "Explain the best model with global SHAP feature importance when available.",
                    "reasoning": "Decision-makers need model behavior translated into feature-level evidence.",
                },
                {
                    "step": 5,
                    "agent": "evaluation",
                    "action": "Evaluate predictive quality and error patterns.",
                    "reasoning": "Diagnostics clarify whether the model is strong enough for the stated goal.",
                },
                {
                    "step": 6,
                    "agent": "reporting",
                    "action": "Create a concise business-facing report with recommendations and exports.",
                    "reasoning": "The analysis should end with actions, not just metrics.",
                },
            ]
            return {
                "status": "success",
                "plan": plan,
                "problem_type": problem_type,
                "detected_target": detected_target,
                "summary": f"Planned a {problem_type} workflow for target '{detected_target}' against the goal: {goal}",
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc), "plan": [], "problem_type": "unknown"}

