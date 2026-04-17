"""LangGraph state graph definition and agent routing.

Defines the end-to-end pipeline as a directed graph where each
node is an agent's ``run(state) -> state`` method. The graph is
compiled into a LangGraph ``CompiledGraph`` that can be invoked
with a single call.
"""

import logging
from typing import Dict, Any, TypedDict, Optional

import pandas as pd
from langgraph.graph import StateGraph, END

# Settings import MUST come first — triggers load_dotenv() at module level
from auto_ds_agent.config.settings import get_settings

from auto_ds_agent.agents.planner import PlannerAgent
from auto_ds_agent.agents.data_agent import DataAgent
from auto_ds_agent.agents.eda_agent import EDAAgent
from auto_ds_agent.agents.ml_agent import MLAgent
from auto_ds_agent.agents.evaluator import EvaluatorAgent
from auto_ds_agent.agents.reporter import ReporterAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    """Typed dictionary representing the shared state passed between agents."""

    # Input
    df: Any  # pd.DataFrame — TypedDict doesn't support non-JSON types natively
    user_goal: str
    dataset_summary: dict
    target_col: str

    # Planner
    plan: dict

    # DataAgent
    cleaning_report: dict
    outlier_indices: dict

    # EDAAgent
    eda_summary_stats: dict
    eda_insights: dict
    eda_plots: dict

    # MLAgent
    ml_plan: dict
    trained_models: dict
    model_results: dict
    model_paths: dict
    best_model_name: str
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    scaler: Any

    # EvaluatorAgent
    cv_results: dict
    feature_importances: dict
    evaluation_report: dict

    # ReporterAgent
    final_report: dict
    final_report_md: str
    report_path: str

    # Errors
    ml_error: str


# ---------------------------------------------------------------------------
# Node wrappers
# ---------------------------------------------------------------------------

def _planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the execution plan."""
    logger.info(">>> Node: planner")
    agent = PlannerAgent()
    dataset_summary = state.get("dataset_summary", {})
    user_goal = state.get("user_goal", "")
    plan = agent.generate_plan(dataset_summary, user_goal)
    state["plan"] = plan
    return state


def _data_cleaning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and preprocess the raw DataFrame."""
    logger.info(">>> Node: data_cleaning")
    settings = get_settings()
    agent = DataAgent(cardinality_threshold=settings.cardinality_threshold)
    return agent.run(state)


def _eda_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Perform exploratory data analysis."""
    logger.info(">>> Node: eda")
    settings = get_settings()
    agent = EDAAgent(output_dir=settings.outputs_dir)
    return agent.run(state)


def _ml_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Train and evaluate ML models."""
    logger.info(">>> Node: ml")
    settings = get_settings()
    agent = MLAgent(saved_models_dir=settings.saved_models_dir)
    return agent.run(state)


def _evaluation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Deeply evaluate trained models."""
    logger.info(">>> Node: evaluation")
    agent = EvaluatorAgent()
    return agent.run(state)


def _report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the final report."""
    logger.info(">>> Node: report")
    settings = get_settings()
    agent = ReporterAgent(output_dir=settings.outputs_dir)
    return agent.run(state)


# ---------------------------------------------------------------------------
# Conditional edge: skip ML if plan says "unknown" or an error occurred
# ---------------------------------------------------------------------------

def _should_run_ml(state: Dict[str, Any]) -> str:
    """Decide whether to proceed to ML training or skip to report."""
    problem_type = state.get("plan", {}).get("problem_type", "unknown")
    if problem_type == "unknown":
        logger.warning("Problem type is 'unknown'. Skipping ML and evaluation.")
        return "report"
    return "ml"


def _should_run_evaluation(state: Dict[str, Any]) -> str:
    """Skip evaluation if ML encountered an error."""
    if state.get("ml_error"):
        logger.warning("ML error detected. Skipping evaluation.")
        return "report"
    return "evaluation"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """Construct and compile the LangGraph pipeline.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready to ``.invoke(state)``.
    """
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("planner", _planner_node)
    graph.add_node("data_cleaning", _data_cleaning_node)
    graph.add_node("eda", _eda_node)
    graph.add_node("ml", _ml_node)
    graph.add_node("evaluation", _evaluation_node)
    graph.add_node("report", _report_node)

    # Define edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "data_cleaning")
    graph.add_edge("data_cleaning", "eda")
    graph.add_conditional_edges("eda", _should_run_ml, {"ml": "ml", "report": "report"})
    graph.add_conditional_edges("ml", _should_run_evaluation, {"evaluation": "evaluation", "report": "report"})
    graph.add_edge("evaluation", "report")
    graph.add_edge("report", END)

    compiled = graph.compile()
    logger.info("Pipeline graph compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    user_goal: str = "",
    target_col: str = "",
) -> Dict[str, Any]:
    """One-call convenience function to run the full pipeline.

    Args:
        df: Raw dataset as a pandas DataFrame.
        user_goal: Optional natural-language goal.
        target_col: Optional explicit target column name.

    Returns:
        Final state dict with all outputs.
    """
    # Build a lightweight dataset summary for the planner
    dataset_summary = {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(3).to_dict(),
        "missing_total": int(df.isna().sum().sum()),
    }

    initial_state: Dict[str, Any] = {
        "df": df,
        "user_goal": user_goal,
        "target_col": target_col,
        "dataset_summary": dataset_summary,
    }

    pipeline = build_graph()
    final_state = pipeline.invoke(initial_state)
    return final_state
