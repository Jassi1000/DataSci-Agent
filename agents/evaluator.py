"""Evaluator agent to assess and validate model performance.

Runs deeper evaluation on the best model(s): cross-validation,
feature importance, and LLM-generated performance interpretation.
"""

import os
import json
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from auto_ds_agent.tools.ml_tools import cross_validate_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ModelVerdict(BaseModel):
    """LLM verdict on a trained model's performance."""

    model_name: str = Field(description="Name of the model evaluated")
    quality: str = Field(description="poor | acceptable | good | excellent")
    explanation: str = Field(description="Plain-English explanation of performance")
    risks: List[str] = Field(default_factory=list, description="Potential risks or caveats")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class EvaluationReport(BaseModel):
    """Full evaluation report across all trained models."""

    verdicts: List[ModelVerdict] = Field(description="Per-model verdicts")
    overall_recommendation: str = Field(
        description="Final recommendation: which model to deploy and why"
    )


# ---------------------------------------------------------------------------
# EvaluatorAgent
# ---------------------------------------------------------------------------

class EvaluatorAgent:
    """Agent that deeply evaluates trained models and produces verdicts.

    Compatible with LangGraph node interface (``run(state) -> state``).
    """

    def __init__(
        self,
        model_name: str = "llama3-70b-8192",
        temperature: float = 0.1,
    ) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set. Evaluation narrative may fail.")

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )
        self.parser = JsonOutputParser(pydantic_object=EvaluationReport)
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm | self.parser

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self) -> PromptTemplate:
        template = """You are a senior ML engineer reviewing model evaluation results.

Problem Type: {problem_type}
User Goal: {user_goal}

Model Results:
{model_results}

Cross-Validation Results:
{cv_results}

Feature Importances (if available):
{feature_importances}

Instructions:
1. For EACH model, provide a verdict: quality rating, plain-English explanation, risks, and suggestions.
2. Provide an overall recommendation on which model to deploy and why.
3. Consider overfitting signals (train vs. CV gap), metric quality, and practical deployment concerns.
4. Return ONLY valid JSON. No markdown.

{format_instructions}
"""
        return PromptTemplate(
            template=template,
            input_variables=[
                "problem_type", "user_goal", "model_results",
                "cv_results", "feature_importances",
            ],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    # ------------------------------------------------------------------
    # Feature importance extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_feature_importances(
        trained_models: Dict[str, Any],
        feature_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Extract feature importances from tree-based models."""

        importances: Dict[str, Dict[str, float]] = {}
        for name, model in trained_models.items():
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                sorted_pairs = sorted(
                    zip(feature_names, imp.tolist()), key=lambda x: x[1], reverse=True
                )
                importances[name] = {feat: round(val, 4) for feat, val in sorted_pairs[:15]}
            elif hasattr(model, "coef_"):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = np.mean(np.abs(coef), axis=0)
                else:
                    coef = np.abs(coef)
                sorted_pairs = sorted(
                    zip(feature_names, coef.tolist()), key=lambda x: x[1], reverse=True
                )
                importances[name] = {feat: round(val, 4) for feat, val in sorted_pairs[:15]}
        return importances

    # ------------------------------------------------------------------
    # LLM evaluation narrative
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        problem_type: str,
        user_goal: str,
        model_results: Dict[str, Any],
        cv_results: Dict[str, Any],
        feature_importances: Dict[str, Any],
        retries: int = 3,
    ) -> EvaluationReport:
        for attempt in range(retries):
            try:
                logger.info("Generating evaluation report. Attempt %d/%d", attempt + 1, retries)
                raw = self.chain.invoke(
                    {
                        "problem_type": problem_type,
                        "user_goal": user_goal or "Build the best model",
                        "model_results": json.dumps(model_results, indent=2, default=str),
                        "cv_results": json.dumps(cv_results, indent=2, default=str),
                        "feature_importances": json.dumps(feature_importances, indent=2, default=str),
                    }
                )
                report = EvaluationReport(**raw)
                logger.info("Evaluation report validated. %d verdicts.", len(report.verdicts))
                return report
            except (ValidationError, Exception) as exc:
                logger.error("Evaluation report attempt %d failed: %s", attempt + 1, exc)

        # Fallback
        logger.warning("LLM retries exhausted. Returning minimal evaluation report.")
        return EvaluationReport(
            verdicts=[
                ModelVerdict(
                    model_name="unknown",
                    quality="acceptable",
                    explanation="LLM evaluation unavailable. Review metrics manually.",
                    risks=["Automated interpretation was not possible."],
                    suggestions=["Manually inspect confusion matrix and residuals."],
                )
            ],
            overall_recommendation="Review model metrics manually.",
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Deeply evaluate trained models.

        Expected state keys (input):
            - trained_models: dict {name: fitted_estimator}
            - model_results: dict  {name: metrics}
            - plan: dict
            - X_train, X_test, y_train, y_test (for supervised)
            - user_goal: str (optional)

        Updated / added state keys (output):
            - cv_results: dict
            - feature_importances: dict
            - evaluation_report: dict
        """
        trained_models = state.get("trained_models", {})
        model_results = state.get("model_results", {})
        problem_type = state.get("plan", {}).get("problem_type", "classification")
        user_goal = state.get("user_goal", "")

        logger.info("EvaluatorAgent started. Evaluating %d models.", len(trained_models))

        # 1. Cross-validation (supervised only)
        cv_results: Dict[str, Any] = {}
        if problem_type in ("classification", "regression"):
            X_train = state.get("X_train")
            y_train = state.get("y_train")
            if X_train is not None and y_train is not None:
                scoring = "f1_weighted" if problem_type == "classification" else "r2"
                for name, model in trained_models.items():
                    try:
                        cv_results[name] = cross_validate_model(
                            model, X_train, y_train, cv=5, scoring=scoring,
                        )
                    except Exception as exc:
                        logger.error("Cross-validation failed for '%s': %s", name, exc)
                        cv_results[name] = {"error": str(exc)}

        state["cv_results"] = cv_results

        # 2. Feature importances
        feature_names = []
        if state.get("X_train") is not None:
            feature_names = state["X_train"].columns.tolist()
        feature_importances = self._extract_feature_importances(trained_models, feature_names)
        state["feature_importances"] = feature_importances

        # 3. LLM evaluation narrative
        eval_report = self._generate_report(
            problem_type, user_goal, model_results, cv_results, feature_importances
        )
        state["evaluation_report"] = eval_report.model_dump()

        logger.info("EvaluatorAgent finished.")
        return state
