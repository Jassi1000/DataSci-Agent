"""LangGraph state graph definition and agent routing.

Defines the end-to-end pipeline as a directed graph where each
node is an agent's ``run(state) -> state`` method.
"""

import logging
from typing import Dict, Any, TypedDict

import pandas as pd
from langgraph.graph import StateGraph, END

# Local imports
from config.settings import get_settings

from agents.planner import PlannerAgent
from agents.data_agent import DataAgent
from agents.eda_agent import EDAAgent
from agents.ml_agent import MLAgent
from agents.evaluator import EvaluatorAgent
from agents.reporter import ReporterAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    df: Any
    user_goal: str
    dataset_summary: dict
    target_col: str

    plan: dict

    cleaning_report: dict
    outlier_indices: dict

    eda_summary_stats: dict
    eda_insights: dict
    eda_plots: dict

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

    cv_results: dict
    feature_importances: dict
    evaluation_report: dict

    final_report: dict
    final_report_md: str
    report_path: str

    ml_error: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Planner Node")
    agent = PlannerAgent()

    plan = agent.generate_plan(
        state.get("dataset_summary", {}),
        state.get("user_goal", "")
    )

    state["plan"] = plan
    return state


def _data_cleaning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Data Cleaning Node")
    settings = get_settings()
    agent = DataAgent(
        cardinality_threshold=settings.cardinality_threshold
    )
    return agent.run(state)


def _eda_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("EDA Node")
    settings = get_settings()
    agent = EDAAgent(output_dir=settings.outputs_dir)
    return agent.run(state)


def _ml_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("ML Node")
    settings = get_settings()
    agent = MLAgent(saved_models_dir=settings.saved_models_dir)
    return agent.run(state)


def _evaluation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Evaluation Node")
    agent = EvaluatorAgent()
    return agent.run(state)


def _report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Report Node")
    settings = get_settings()
    agent = ReporterAgent(output_dir=settings.outputs_dir)
    return agent.run(state)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def _should_run_ml(state: Dict[str, Any]) -> str:
    problem_type = state.get("plan", {}).get("problem_type", "unknown")

    if problem_type == "unknown":
        return "report"

    return "ml"


def _should_run_evaluation(state: Dict[str, Any]) -> str:
    if state.get("ml_error"):
        return "report"

    return "evaluation"


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(dict)

    graph.add_node("planner", _planner_node)
    graph.add_node("data_cleaning", _data_cleaning_node)
    graph.add_node("eda", _eda_node)
    graph.add_node("ml", _ml_node)
    graph.add_node("evaluation", _evaluation_node)
    graph.add_node("report", _report_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "data_cleaning")
    graph.add_edge("data_cleaning", "eda")

    graph.add_conditional_edges(
        "eda",
        _should_run_ml,
        {
            "ml": "ml",
            "report": "report"
        }
    )

    graph.add_conditional_edges(
        "ml",
        _should_run_evaluation,
        {
            "evaluation": "evaluation",
            "report": "report"
        }
    )

    graph.add_edge("evaluation", "report")
    graph.add_edge("report", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    user_goal: str = "",
    target_col: str = ""
) -> Dict[str, Any]:

    dataset_summary = {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {
            col: str(dtype) for col, dtype in df.dtypes.items()
        },
        "sample": df.head(3).to_dict(),
        "missing_total": int(df.isna().sum().sum()),
    }

    initial_state = {
        "df": df,
        "user_goal": user_goal,
        "target_col": target_col,
        "dataset_summary": dataset_summary,
    }

    pipeline = build_graph()
    return pipeline.invoke(initial_state)