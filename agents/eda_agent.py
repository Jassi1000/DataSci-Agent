"""EDA agent for exploratory data analysis and statistics.

Generates summary statistics, visualisations, and LLM-powered
narrative insights about the dataset.
"""

import os
import json
import logging
from typing import Dict, Any, List

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from tools.viz_tools import (
    plot_correlation_matrix,
    plot_distribution,
    plot_boxplots,
    plot_missing_heatmap,
    generate_summary_stats,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class EDAInsight(BaseModel):
    """A single insight produced by the LLM about the dataset."""

    topic: str = Field(description="Short topic label, e.g. 'Target Imbalance'")
    description: str = Field(description="Detailed explanation of the insight")
    severity: str = Field(description="low | medium | high — impact on modelling")


class EDAReport(BaseModel):
    """Structured EDA report returned by the LLM."""

    insights: List[EDAInsight] = Field(description="List of analytical insights")
    recommended_target: str = Field(
        default="",
        description="Suggested target column name (empty string if unclear)",
    )
    feature_notes: List[str] = Field(
        default_factory=list,
        description="Per-feature observations worth noting",
    )


# ---------------------------------------------------------------------------
# EDAAgent
# ---------------------------------------------------------------------------

class EDAAgent:
    """Agent that performs exploratory data analysis on a cleaned DataFrame.

    Compatible with LangGraph node interface (``run(state) -> state``).
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        output_dir: str = "storage/outputs",
    ) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set. LLM insight generation may fail.")

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.parser = JsonOutputParser(pydantic_object=EDAReport)
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm | self.parser

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self) -> PromptTemplate:
        template = """You are an expert Data Scientist performing Exploratory Data Analysis.

Below is the statistical summary of a cleaned dataset.

Summary Statistics:
{summary_stats}

Cleaning Report (from previous agent):
{cleaning_report}

User Goal: {user_goal}

Instructions:
1. Identify the most important insights (correlations, skewness, class imbalance, dominant features, etc.).
2. Suggest which column is the most likely TARGET variable based on column names and statistics.
3. Note any feature-level observations that could affect modelling.
4. Classify each insight by severity (low / medium / high).
5. Return ONLY valid JSON. No markdown code blocks.

{format_instructions}
"""
        return PromptTemplate(
            template=template,
            input_variables=["summary_stats", "cleaning_report", "user_goal"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    # ------------------------------------------------------------------
    # LLM insight generation
    # ------------------------------------------------------------------

    def _generate_insights(
        self,
        summary_stats: Dict[str, Any],
        cleaning_report: Dict[str, Any],
        user_goal: str,
        retries: int = 3,
    ) -> EDAReport:
        """Ask the LLM for narrative insights based on summary stats."""

        for attempt in range(retries):
            try:
                logger.info("Generating EDA insights. Attempt %d/%d", attempt + 1, retries)
                raw = self.chain.invoke(
                    {
                        "summary_stats": json.dumps(summary_stats, indent=2, default=str),
                        "cleaning_report": json.dumps(cleaning_report, indent=2, default=str),
                        "user_goal": user_goal or "General exploration",
                    }
                )
                report = EDAReport(**raw)
                logger.info("EDA insights validated. %d insights generated.", len(report.insights))
                return report
            except (ValidationError, Exception) as exc:
                logger.error("EDA insight attempt %d failed: %s", attempt + 1, exc)

        logger.warning("LLM retries exhausted. Returning minimal EDA report.")
        return EDAReport(
            insights=[
                EDAInsight(
                    topic="Fallback",
                    description="LLM-based insights unavailable. Review plots manually.",
                    severity="medium",
                )
            ],
            recommended_target="",
            feature_notes=[],
        )

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _generate_plots(self, df: pd.DataFrame) -> Dict[str, str]:
        """Create standard EDA plots and return paths."""

        paths: Dict[str, str] = {}

        paths["correlation_matrix"] = plot_correlation_matrix(
            df, os.path.join(self.output_dir, "correlation_matrix.png")
        )
        paths["boxplots"] = plot_boxplots(
            df, os.path.join(self.output_dir, "boxplots.png")
        )
        paths["missing_heatmap"] = plot_missing_heatmap(
            df, os.path.join(self.output_dir, "missing_heatmap.png")
        )

        # Individual distributions for first N columns
        numeric_cols = df.select_dtypes(include=["number"]).columns[:10].tolist()
        for col in numeric_cols:
            safe_name = col.replace(" ", "_").replace("/", "_")
            path = plot_distribution(
                df[col], col, os.path.join(self.output_dir, f"dist_{safe_name}.png")
            )
            paths[f"dist_{col}"] = path

        return {k: v for k, v in paths.items() if v}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute EDA on the cleaned DataFrame.

        Expected state keys (input):
            - df: pd.DataFrame (cleaned)
            - cleaning_report: dict
            - user_goal: str (optional)

        Updated / added state keys (output):
            - eda_summary_stats: dict
            - eda_insights: dict (EDAReport serialised)
            - eda_plots: dict  (name → file path)
        """
        df: pd.DataFrame = state["df"]
        cleaning_report = state.get("cleaning_report", {})
        user_goal = state.get("user_goal", "")

        logger.info("EDAAgent started. Shape: %s", df.shape)

        # 1. Summary statistics
        summary_stats = generate_summary_stats(df)
        state["eda_summary_stats"] = summary_stats

        # 2. Plots
        plots = self._generate_plots(df)
        state["eda_plots"] = plots
        logger.info("Generated %d EDA plots.", len(plots))

        # 3. LLM insights
        eda_report = self._generate_insights(summary_stats, cleaning_report, user_goal)
        state["eda_insights"] = eda_report.model_dump()

        # If the LLM suggested a target and one isn't set yet, propagate it
        if eda_report.recommended_target and not state.get("target_col"):
            state["target_col"] = eda_report.recommended_target
            logger.info("EDAAgent inferred target column: '%s'", eda_report.recommended_target)

        logger.info("EDAAgent finished.")
        return state
