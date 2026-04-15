"""Data agent handling data loading, cleaning, and preprocessing.

Uses deterministic tools from tools.data_tools for heavy lifting
and delegates ambiguous decisions (e.g. imputation strategy choice)
to an LLM via LangChain + Groq.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from tools.data_tools import (
    get_missing_summary,
    impute_column,
    detect_outliers_iqr,
    fix_dtypes,
    encode_categoricals,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class ImputationDecision(BaseModel):
    """LLM-suggested imputation strategy for a single column."""

    column: str = Field(description="Column name")
    strategy: str = Field(description="Imputation strategy: mean | median | mode | drop")
    reasoning: str = Field(description="Brief reasoning for the chosen strategy")


class ImputationPlan(BaseModel):
    """Collection of imputation decisions returned by the LLM."""

    decisions: List[ImputationDecision] = Field(
        description="List of per-column imputation decisions"
    )


# ---------------------------------------------------------------------------
# DataAgent
# ---------------------------------------------------------------------------

class DataAgent:
    """Agent responsible for cleaning, transforming, and preparing raw data.

    Designed to be invoked by the orchestrator with a shared ``state`` dict.
    Compatible with LangGraph node interface (``run(state) -> state``).
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        cardinality_threshold: int = 10,
    ) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY environment variable is not set. LLM calls may fail.")

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )
        self.cardinality_threshold = cardinality_threshold
        self.parser = JsonOutputParser(pydantic_object=ImputationPlan)
        self.prompt = self._build_imputation_prompt()
        self.chain = self.prompt | self.llm | self.parser

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_imputation_prompt(self) -> PromptTemplate:
        """Construct the prompt used to ask the LLM for imputation strategies."""

        template = """You are an expert Data Scientist.
Given the missing-value summary below, decide the best imputation strategy for EACH column that has missing values.

Missing Value Summary:
{missing_summary}

Dataset shape: {shape}
User goal (if any): {user_goal}

Rules:
- For numeric columns with < 5% missing → use "mean" or "median" (prefer median if outliers likely).
- For numeric columns with 5-40% missing → use "median".
- For numeric columns with > 40% missing → use "drop" (the column provides little value).
- For categorical columns with < 30% missing → use "mode".
- For categorical columns with ≥ 30% missing → use "drop".
- For columns that are entirely null → use "drop".
- Always provide brief reasoning.

Return ONLY valid JSON adhering to the format instructions below. No markdown.

Format Instructions:
{format_instructions}
"""
        return PromptTemplate(
            template=template,
            input_variables=["missing_summary", "shape", "user_goal"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    # ------------------------------------------------------------------
    # LLM-backed imputation planning
    # ------------------------------------------------------------------

    def _get_imputation_plan(
        self,
        missing_summary: Dict[str, Any],
        shape: tuple,
        user_goal: str,
        retries: int = 3,
    ) -> List[ImputationDecision]:
        """Ask the LLM which imputation strategy to use per column.

        Falls back to a deterministic heuristic if the LLM fails after
        ``retries`` attempts.
        """
        # Filter to only columns that actually have missing values
        cols_with_missing = {
            col: info
            for col, info in missing_summary.items()
            if info["missing_count"] > 0
        }

        if not cols_with_missing:
            logger.info("No missing values detected. Skipping imputation planning.")
            return []

        for attempt in range(retries):
            try:
                logger.info("Requesting LLM imputation plan. Attempt %d/%d", attempt + 1, retries)
                raw = self.chain.invoke(
                    {
                        "missing_summary": json.dumps(cols_with_missing, indent=2),
                        "shape": str(shape),
                        "user_goal": user_goal,
                    }
                )
                plan = ImputationPlan(**raw)
                logger.info("LLM imputation plan validated successfully.")
                return plan.decisions

            except (ValidationError, Exception) as exc:
                logger.error("LLM imputation attempt %d failed: %s", attempt + 1, exc)

        # Deterministic fallback
        logger.warning("LLM retries exhausted. Using heuristic imputation fallback.")
        return self._heuristic_imputation(cols_with_missing)

    def _heuristic_imputation(
        self, cols_with_missing: Dict[str, Any]
    ) -> List[ImputationDecision]:
        """Deterministic fallback when the LLM is unavailable."""

        decisions: List[ImputationDecision] = []
        for col, info in cols_with_missing.items():
            pct = info["missing_pct"]
            dtype = info["dtype"]
            is_numeric = dtype in (
                "int64", "int32", "float64", "float32",
                "Int64", "Int32", "Float64", "Float32",
            )

            if pct > 40:
                strategy = "drop"
                reason = f"{pct}% missing — too sparse to impute reliably."
            elif is_numeric:
                strategy = "median"
                reason = f"Numeric column with {pct}% missing — median is robust to outliers."
            else:
                if pct >= 30:
                    strategy = "drop"
                    reason = f"Categorical column with {pct}% missing — dropping."
                else:
                    strategy = "mode"
                    reason = f"Categorical column with {pct}% missing — imputing with mode."

            decisions.append(
                ImputationDecision(column=col, strategy=strategy, reasoning=reason)
            )
        return decisions

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full data-cleaning pipeline.

        Expected state keys (input):
            - df: pd.DataFrame  — raw dataset
            - plan: dict        — ExecutionPlan from PlannerAgent
            - user_goal: str    — (optional) original user goal

        Updated / added state keys (output):
            - df: pd.DataFrame          — cleaned dataset
            - cleaning_report: dict     — summary of all actions taken
        """
        df: pd.DataFrame = state["df"].copy()
        user_goal: str = state.get("user_goal", "")
        report: Dict[str, Any] = {}

        logger.info("DataAgent started. Initial shape: %s", df.shape)
        report["initial_shape"] = list(df.shape)

        # 1. Fix dtypes -------------------------------------------------------
        logger.info("Step 1/5 — Fixing dtypes.")
        df = fix_dtypes(df)
        report["dtypes_fixed"] = True

        # 2. Remove duplicates -------------------------------------------------
        logger.info("Step 2/5 — Removing duplicates.")
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            logger.info("Removed %d duplicate rows.", dup_count)
        report["duplicates_removed"] = dup_count

        # 3. Handle missing values ---------------------------------------------
        logger.info("Step 3/5 — Handling missing values.")
        missing_summary = get_missing_summary(df)
        report["missing_summary_before"] = missing_summary

        imputation_decisions = self._get_imputation_plan(
            missing_summary, df.shape, user_goal
        )

        imputation_log: List[Dict[str, str]] = []
        for decision in imputation_decisions:
            col = decision.column
            strategy = decision.strategy

            if col not in df.columns:
                logger.warning("Column '%s' from imputation plan not in DataFrame. Skipping.", col)
                continue

            if strategy == "drop":
                # Drop entire column when the LLM / heuristic says "drop"
                df = df.drop(columns=[col])
                logger.info("Dropped column '%s' (%.1f%% missing).", col, missing_summary[col]["missing_pct"])
            else:
                df = impute_column(df, col, strategy)

            imputation_log.append(
                {"column": col, "strategy": strategy, "reasoning": decision.reasoning}
            )

        report["imputation_actions"] = imputation_log

        # 4. Detect outliers (flag only) ----------------------------------------
        logger.info("Step 4/5 — Detecting outliers (IQR).")
        outlier_report: Dict[str, List[int]] = {}
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        for col in numeric_cols:
            result = detect_outliers_iqr(df, col)
            if result[col]:
                outlier_report[col] = result[col]

        report["outliers_flagged"] = {
            col: len(indices) for col, indices in outlier_report.items()
        }
        # Store raw indices in state for downstream agents if needed
        state["outlier_indices"] = outlier_report

        # 5. Encode categoricals ------------------------------------------------
        logger.info("Step 5/5 — Encoding categorical columns.")
        cat_cols_before = df.select_dtypes(include=["object", "category"]).columns.tolist()
        df = encode_categoricals(df, threshold=self.cardinality_threshold)
        cat_cols_after = df.select_dtypes(include=["object", "category"]).columns.tolist()
        report["categoricals_encoded"] = {
            "before": cat_cols_before,
            "remaining_object_cols": cat_cols_after,
        }

        # Finalise -------------------------------------------------------------
        report["final_shape"] = list(df.shape)
        logger.info("DataAgent finished. Final shape: %s", df.shape)

        state["df"] = df
        state["cleaning_report"] = report
        return state
