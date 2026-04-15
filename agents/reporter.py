"""Reporter agent to compile findings into final summaries.

Collects outputs from every prior agent and uses the LLM to
produce a polished, human-readable Markdown report.
"""

import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime, timezone

from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ReportSection(BaseModel):
    """A single section of the final report."""

    title: str = Field(description="Section heading")
    content: str = Field(description="Markdown-formatted section body")


class FinalReport(BaseModel):
    """Structured final report returned by the LLM."""

    title: str = Field(description="Report title")
    executive_summary: str = Field(description="2-3 sentence executive summary")
    sections: List[ReportSection] = Field(description="Ordered list of report sections")
    conclusion: str = Field(description="Final conclusion and next steps")


# ---------------------------------------------------------------------------
# ReporterAgent
# ---------------------------------------------------------------------------

class ReporterAgent:
    """Agent that compiles all pipeline outputs into a final report.

    Compatible with LangGraph node interface (``run(state) -> state``).
    """

    def __init__(
        self,
        model_name: str = "llama3-70b-8192",
        temperature: float = 0.2,
        output_dir: str = "storage/outputs",
    ) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set. Report generation may fail.")

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.parser = JsonOutputParser(pydantic_object=FinalReport)
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm | self.parser

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self) -> PromptTemplate:
        template = """You are a senior Data Scientist writing a final analysis report.

Compile the information below into a professional, well-structured report in Markdown.

User Goal: {user_goal}
Problem Type: {problem_type}

Data Cleaning Summary:
{cleaning_report}

EDA Insights:
{eda_insights}

Model Results:
{model_results}

Cross-Validation Results:
{cv_results}

Feature Importances:
{feature_importances}

Evaluation Verdicts:
{evaluation_report}

Best Model: {best_model_name}

Instructions:
1. Write a clear executive summary.
2. Include sections: Data Overview, Cleaning & Preprocessing, Exploratory Analysis,
   Modelling, Evaluation, and Recommendations.
3. Use bullet points and tables (Markdown format) where appropriate.
4. Keep the language professional but accessible.
5. Return ONLY valid JSON adhering to the format instructions. No markdown code fences.

{format_instructions}
"""
        return PromptTemplate(
            template=template,
            input_variables=[
                "user_goal", "problem_type", "cleaning_report", "eda_insights",
                "model_results", "cv_results", "feature_importances",
                "evaluation_report", "best_model_name",
            ],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    # ------------------------------------------------------------------
    # LLM report generation
    # ------------------------------------------------------------------

    def _generate_report(self, state: Dict[str, Any], retries: int = 3) -> FinalReport:
        """Ask the LLM to produce the final Markdown report."""

        def _safe_json(obj: Any) -> str:
            return json.dumps(obj, indent=2, default=str)

        for attempt in range(retries):
            try:
                logger.info("Generating final report. Attempt %d/%d", attempt + 1, retries)
                raw = self.chain.invoke(
                    {
                        "user_goal": state.get("user_goal", "General analysis"),
                        "problem_type": state.get("plan", {}).get("problem_type", "unknown"),
                        "cleaning_report": _safe_json(state.get("cleaning_report", {})),
                        "eda_insights": _safe_json(state.get("eda_insights", {})),
                        "model_results": _safe_json(state.get("model_results", {})),
                        "cv_results": _safe_json(state.get("cv_results", {})),
                        "feature_importances": _safe_json(state.get("feature_importances", {})),
                        "evaluation_report": _safe_json(state.get("evaluation_report", {})),
                        "best_model_name": state.get("best_model_name", "N/A"),
                    }
                )
                report = FinalReport(**raw)
                logger.info("Final report validated. %d sections.", len(report.sections))
                return report
            except (ValidationError, Exception) as exc:
                logger.error("Report generation attempt %d failed: %s", attempt + 1, exc)

        logger.warning("LLM retries exhausted. Returning fallback report.")
        return FinalReport(
            title="Auto DS Agent — Analysis Report",
            executive_summary="Automated report generation failed. Please review raw outputs.",
            sections=[
                ReportSection(
                    title="Raw Outputs",
                    content="Refer to `state` keys: cleaning_report, eda_insights, model_results, evaluation_report.",
                )
            ],
            conclusion="Manual review required.",
        )

    # ------------------------------------------------------------------
    # Markdown serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_markdown(report: FinalReport) -> str:
        """Convert a FinalReport into a Markdown string."""

        lines: List[str] = []
        lines.append(f"# {report.title}\n")
        lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
        lines.append("---\n")
        lines.append(f"## Executive Summary\n\n{report.executive_summary}\n")

        for section in report.sections:
            lines.append(f"## {section.title}\n\n{section.content}\n")

        lines.append(f"## Conclusion\n\n{report.conclusion}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compile the final report from all pipeline outputs.

        Expected state keys (input):
            All keys populated by prior agents.

        Updated / added state keys (output):
            - final_report: dict     (FinalReport serialised)
            - final_report_md: str   (Markdown string)
            - report_path: str       (saved file path)
        """
        logger.info("ReporterAgent started.")

        report = self._generate_report(state)
        md_content = self._to_markdown(report)

        # Persist to file
        report_path = os.path.join(self.output_dir, "final_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Final report saved to %s", report_path)

        state["final_report"] = report.model_dump()
        state["final_report_md"] = md_content
        state["report_path"] = os.path.abspath(report_path)

        logger.info("ReporterAgent finished.")
        return state
