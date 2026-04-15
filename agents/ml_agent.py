"""ML agent managing model selection, training, and tuning.

Uses LLM reasoning (Groq) to decide which models to train and what
hyperparameter strategy to follow, then delegates the actual
training to tools/ml_tools.py.
"""

import os
import json
import logging
import pickle
from typing import Dict, Any, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from auto_ds_agent.tools.ml_tools import (
    get_model_registry,
    split_data,
    scale_features,
    train_model,
    evaluate_classification,
    evaluate_regression,
    evaluate_clustering,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class ModelSelection(BaseModel):
    """LLM recommendation for a single model to train."""

    model_name: str = Field(description="Key from the model registry, e.g. 'random_forest_classifier'")
    reasoning: str = Field(description="Why this model was selected")


class MLPlan(BaseModel):
    """LLM-generated model training plan."""

    target_column: str = Field(description="Name of the target column to predict")
    models_to_train: List[ModelSelection] = Field(description="Ordered list of models to train")
    scale_features: bool = Field(
        default=True,
        description="Whether to apply feature scaling before training",
    )
    notes: str = Field(default="", description="Any additional notes for modelling")


# ---------------------------------------------------------------------------
# MLAgent
# ---------------------------------------------------------------------------

class MLAgent:
    """Agent that selects, trains, and evaluates ML models.

    Compatible with LangGraph node interface (``run(state) -> state``).
    """

    def __init__(
        self,
        model_name: str = "llama3-70b-8192",
        temperature: float = 0.1,
        saved_models_dir: str = "models/saved_models",
    ) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set. LLM model-selection may fail.")

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )
        self.saved_models_dir = saved_models_dir
        os.makedirs(self.saved_models_dir, exist_ok=True)

        self.parser = JsonOutputParser(pydantic_object=MLPlan)
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm | self.parser

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self) -> PromptTemplate:
        template = """You are an expert Machine Learning Engineer.

Based on the information below, decide:
1. Which TARGET column to use.
2. Which models to train (pick 2-4 from the available registry).
3. Whether feature scaling is needed.

Problem Type: {problem_type}

Available Models:
{available_models}

EDA Insights:
{eda_insights}

Dataset columns: {columns}
Dataset shape: {shape}
User Goal: {user_goal}

Rules:
- Pick models from the available list ONLY.
- Prefer diverse models (e.g. one linear, one tree-based, one ensemble).
- If the problem type is 'clustering', set target_column to an empty string.
- Return ONLY valid JSON. No markdown.

{format_instructions}
"""
        return PromptTemplate(
            template=template,
            input_variables=[
                "problem_type", "available_models", "eda_insights",
                "columns", "shape", "user_goal",
            ],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    # ------------------------------------------------------------------
    # LLM-backed model selection
    # ------------------------------------------------------------------

    def _get_ml_plan(
        self,
        problem_type: str,
        df: pd.DataFrame,
        eda_insights: Dict[str, Any],
        user_goal: str,
        retries: int = 3,
    ) -> MLPlan:
        registry = get_model_registry(problem_type)
        available = list(registry.keys())

        for attempt in range(retries):
            try:
                logger.info("Requesting ML plan. Attempt %d/%d", attempt + 1, retries)
                raw = self.chain.invoke(
                    {
                        "problem_type": problem_type,
                        "available_models": json.dumps(available),
                        "eda_insights": json.dumps(eda_insights, indent=2, default=str),
                        "columns": json.dumps(df.columns.tolist()),
                        "shape": str(df.shape),
                        "user_goal": user_goal or "Build the best baseline model",
                    }
                )
                plan = MLPlan(**raw)
                # Validate model names against registry
                plan.models_to_train = [
                    m for m in plan.models_to_train if m.model_name in registry
                ]
                if not plan.models_to_train:
                    raise ValueError("LLM returned no valid model names from registry.")
                logger.info("ML plan validated. %d models selected.", len(plan.models_to_train))
                return plan
            except (ValidationError, ValueError, Exception) as exc:
                logger.error("ML plan attempt %d failed: %s", attempt + 1, exc)

        # Fallback: pick first two models
        logger.warning("LLM retries exhausted. Using fallback ML plan.")
        fallback_models = available[:2] if len(available) >= 2 else available
        return MLPlan(
            target_column=self._guess_target(df),
            models_to_train=[
                ModelSelection(model_name=m, reasoning="Fallback selection")
                for m in fallback_models
            ],
            scale_features=True,
            notes="Fallback plan — LLM was unavailable.",
        )

    @staticmethod
    def _guess_target(df: pd.DataFrame) -> str:
        """Heuristic: last column is often the target."""
        return df.columns[-1] if not df.columns.empty else ""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model(self, model: Any, name: str) -> str:
        path = os.path.join(self.saved_models_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model '%s' saved to %s", name, path)
        return os.path.abspath(path)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Train and evaluate models based on LLM-generated plan.

        Expected state keys (input):
            - df: pd.DataFrame (cleaned + encoded)
            - plan: dict (ExecutionPlan from PlannerAgent)
            - eda_insights: dict
            - target_col: str (optional, may be set by EDAAgent)
            - user_goal: str (optional)

        Updated / added state keys (output):
            - ml_plan: dict
            - trained_models: dict  {model_name: fitted_estimator}
            - model_results: dict   {model_name: evaluation_metrics}
            - model_paths: dict     {model_name: saved_file_path}
            - best_model_name: str
            - X_train, X_test, y_train, y_test (for downstream agents)
        """
        df: pd.DataFrame = state["df"]
        problem_type: str = state.get("plan", {}).get("problem_type", "classification")
        eda_insights = state.get("eda_insights", {})
        user_goal = state.get("user_goal", "")

        logger.info("MLAgent started. Problem type: %s, shape: %s", problem_type, df.shape)

        # 1. Get ML plan from LLM
        ml_plan = self._get_ml_plan(problem_type, df, eda_insights, user_goal)

        # Allow state-level override of target col
        if state.get("target_col"):
            ml_plan.target_column = state["target_col"]

        state["ml_plan"] = ml_plan.model_dump()
        registry = get_model_registry(problem_type)

        trained_models: Dict[str, Any] = {}
        model_results: Dict[str, Any] = {}
        model_paths: Dict[str, str] = {}

        # 2. Prepare data
        if problem_type in ("classification", "regression"):
            target_col = ml_plan.target_column
            if target_col not in df.columns:
                logger.error("Target column '%s' not in DataFrame. Aborting ML.", target_col)
                state["ml_error"] = f"Target column '{target_col}' not found."
                return state

            X_train, X_test, y_train, y_test = split_data(df, target_col)

            if ml_plan.scale_features:
                X_train, X_test, scaler = scale_features(X_train, X_test)
                state["scaler"] = scaler

            state["X_train"] = X_train
            state["X_test"] = X_test
            state["y_train"] = y_train
            state["y_test"] = y_test

            # 3. Train & evaluate each model
            for selection in ml_plan.models_to_train:
                name = selection.model_name
                if name not in registry:
                    logger.warning("Model '%s' not in registry. Skipping.", name)
                    continue

                try:
                    fitted = train_model(registry[name], X_train, y_train)
                    trained_models[name] = fitted
                    model_paths[name] = self._save_model(fitted, name)

                    if problem_type == "classification":
                        model_results[name] = evaluate_classification(fitted, X_test, y_test)
                    else:
                        model_results[name] = evaluate_regression(fitted, X_test, y_test)
                except Exception as exc:
                    logger.error("Training failed for '%s': %s", name, exc)
                    model_results[name] = {"error": str(exc)}

        elif problem_type == "clustering":
            X = df.copy()
            if ml_plan.scale_features:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                state["scaler"] = scaler

            for selection in ml_plan.models_to_train:
                name = selection.model_name
                if name not in registry:
                    continue
                try:
                    fitted = train_model(registry[name], X, None)
                    trained_models[name] = fitted
                    model_paths[name] = self._save_model(fitted, name)
                    model_results[name] = evaluate_clustering(fitted, X)
                except Exception as exc:
                    logger.error("Clustering failed for '%s': %s", name, exc)
                    model_results[name] = {"error": str(exc)}

        # 4. Determine best model
        best_model_name = self._pick_best(problem_type, model_results)

        state["trained_models"] = trained_models
        state["model_results"] = model_results
        state["model_paths"] = model_paths
        state["best_model_name"] = best_model_name

        logger.info("MLAgent finished. Best model: %s", best_model_name)
        return state

    @staticmethod
    def _pick_best(problem_type: str, results: Dict[str, Dict]) -> str:
        """Select the best model name based on the primary metric."""

        metric_map = {
            "classification": "f1_score",
            "regression": "r2_score",
            "clustering": "silhouette_score",
        }
        metric = metric_map.get(problem_type, "f1_score")
        best_name = ""
        best_score = -float("inf")

        for name, metrics in results.items():
            if "error" in metrics:
                continue
            score = metrics.get(metric)
            if score is not None and score > best_score:
                best_score = score
                best_name = name

        return best_name
