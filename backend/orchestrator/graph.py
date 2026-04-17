# backend/orchestrator/graph.py
import json
import time
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from sklearn.model_selection import train_test_split

from agents.data_agent import DataCleaningAgent
from agents.eda_agent import EDAAgent
from agents.evaluator import EvaluatorAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.ml_agent import MLAgent
from agents.planner import PlannerAgent
from agents.reporter import ReporterAgent
from tools.data_tools import dataframe_info
from utils.serializer import make_serializable


class PipelineState(TypedDict, total=False):
    df: Any
    goal: str
    target_col: str
    selected_models: list
    user_params: dict
    fast_mode: bool
    run_explainability: bool
    run_id: str
    dataset_path: str
    dataset_name: str
    created_at: float
    status: str
    plan_results: dict
    cleaning_results: dict
    eda_results: dict
    ml_results: dict
    explainability_results: dict
    evaluation_results: dict
    report_results: dict
    cleaned_df: Any
    problem_type: str


def _notify(callback, agent, status, message):
    if callback:
        callback(agent, status, message)


def _build_graph(progress_callback=None):
    from langgraph.graph import END, StateGraph

    planner = PlannerAgent()
    cleaner = DataCleaningAgent()
    eda_agent = EDAAgent()
    ml_agent = MLAgent()
    explainer = ExplainabilityAgent()
    evaluator = EvaluatorAgent()
    reporter = ReporterAgent()

    def plan_node(state: PipelineState) -> PipelineState:
        _notify(progress_callback, "planner", "running", "Planning analysis workflow...")
        result = planner.run(state["goal"], dataframe_info(state["df"]), state.get("target_col"))
        if result.get("status") == "error":
            raise RuntimeError(result.get("error", "Planner failed."))
        state["plan_results"] = result
        state["problem_type"] = result.get("problem_type", "classification")
        state["target_col"] = state.get("target_col") or result.get("detected_target")
        _notify(progress_callback, "planner", "done", "Analysis plan created.")
        return state

    def clean_node(state: PipelineState) -> PipelineState:
        _notify(progress_callback, "data_cleaner", "running", "Cleaning dataset...")
        result = cleaner.run(state["df"], state.get("target_col"))
        if result.get("status") == "error":
            raise RuntimeError(result.get("error", "Data cleaning failed."))
        state["cleaning_results"] = result
        state["cleaned_df"] = result.get("cleaned_df", state["df"])
        _notify(progress_callback, "data_cleaner", "done", "Dataset cleaned.")
        return state

    def eda_node(state: PipelineState) -> PipelineState:
        _notify(progress_callback, "eda", "running", "Running exploratory analysis...")
        result = eda_agent.run(state["cleaned_df"], state.get("target_col"), state.get("problem_type"))
        if result.get("status") == "error":
            raise RuntimeError(result.get("error", "EDA failed."))
        state["eda_results"] = result
        _notify(progress_callback, "eda", "done", "Exploratory analysis completed.")
        return state

    def ml_node(state: PipelineState) -> PipelineState:
        _notify(progress_callback, "ml_trainer", "running", "Training candidate models...")
        result = ml_agent.run(
            state["cleaned_df"],
            state["target_col"],
            state["problem_type"],
            state.get("selected_models"),
            state.get("user_params"),
            state.get("fast_mode", False),
            None,
            state["run_id"],
        )
        state["ml_results"] = result
        _notify(progress_callback, "ml_trainer", "done" if result.get("status") == "success" else "error", result.get("why_best") or result.get("error", "Model training finished."))
        if result.get("status") == "error":
            raise RuntimeError(result.get("error", "Model training failed."))
        return state

    def explain_node(state: PipelineState) -> PipelineState:
        if not state.get("run_explainability", True):
            state["explainability_results"] = {"status": "skipped", "reason": "Explainability disabled for this run.", "shap_available": False}
            _notify(progress_callback, "explainer", "done", "Explainability skipped.")
            return state
        _notify(progress_callback, "explainer", "running", "Generating model explanations...")
        ml = state.get("ml_results", {})
        if ml.get("status") != "success":
            state["explainability_results"] = {"status": "skipped", "reason": "Model training did not complete."}
        else:
            X = state["cleaned_df"].drop(columns=[state["target_col"]])
            y = state["cleaned_df"][state["target_col"]]
            stratify = y if state["problem_type"] == "classification" and y.nunique() > 1 else None
            X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
            state["explainability_results"] = explainer.run(ml.get("model_path"), X_train, X_test, ml.get("feature_names", X.columns.tolist()), ml.get("best_model_name"), state["problem_type"])
        _notify(progress_callback, "explainer", "done", "Explainability completed.")
        return state

    def evaluate_node(state: PipelineState) -> PipelineState:
        _notify(progress_callback, "evaluator", "running", "Evaluating best model...")
        ml = state.get("ml_results", {})
        state["evaluation_results"] = evaluator.run(ml, ml.get("y_test", []), ml.get("y_pred", []), state.get("problem_type", "classification"))
        _notify(progress_callback, "evaluator", "done", "Evaluation completed.")
        return state

    def report_node(state: PipelineState) -> PipelineState:
        _notify(progress_callback, "reporter", "running", "Writing final report...")
        state["report_results"] = reporter.run(state.get("plan_results", {}), state.get("cleaning_results", {}), state.get("eda_results", {}), state.get("ml_results", {}), state.get("explainability_results", {}), state.get("evaluation_results", {}), state.get("dataset_name", "dataset"))
        state["status"] = "completed"
        _notify(progress_callback, "reporter", "done", "Report generated.")
        return state

    workflow = StateGraph(PipelineState)
    workflow.add_node("planner", plan_node)
    workflow.add_node("data_cleaner", clean_node)
    workflow.add_node("eda", eda_node)
    workflow.add_node("ml_trainer", ml_node)
    workflow.add_node("explainer", explain_node)
    workflow.add_node("evaluator", evaluate_node)
    workflow.add_node("reporter", report_node)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "data_cleaner")
    workflow.add_edge("data_cleaner", "eda")
    workflow.add_edge("eda", "ml_trainer")
    workflow.add_edge("ml_trainer", "explainer")
    workflow.add_edge("explainer", "evaluator")
    workflow.add_edge("evaluator", "reporter")
    workflow.add_edge("reporter", END)
    return workflow.compile()


def run_pipeline(df, goal, target_col, selected_models, user_params, fast_mode, run_id, progress_callback=None, dataset_path=None, dataset_name=None, created_at=None, run_explainability=True) -> dict:
    state = {
        "df": df,
        "goal": goal,
        "target_col": target_col,
        "selected_models": selected_models,
        "user_params": user_params or {},
        "fast_mode": fast_mode,
        "run_explainability": run_explainability,
        "run_id": run_id,
        "dataset_path": dataset_path,
        "dataset_name": dataset_name or "dataset",
        "created_at": created_at or time.time(),
        "status": "running",
    }
    try:
        final_state = _build_graph(progress_callback).invoke(state)
        final_state["dataset_path"] = dataset_path
        final_state["dataset_name"] = dataset_name
        final_state["created_at"] = state["created_at"]
        final_state["goal"] = goal
        final_state["run_id"] = run_id
        serializable = make_serializable({k: v for k, v in final_state.items() if k not in {"df", "cleaned_df"}})
        runs_dir = Path("./storage/runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / f"{run_id}.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        _notify(progress_callback, "pipeline", "completed", "Pipeline completed successfully.")
        return serializable
    except Exception as exc:
        error_state = make_serializable({"run_id": run_id, "status": "failed", "error": str(exc), "dataset_path": dataset_path, "dataset_name": dataset_name, "created_at": state["created_at"], "goal": goal})
        Path("./storage/runs").mkdir(parents=True, exist_ok=True)
        Path(f"./storage/runs/{run_id}.json").write_text(json.dumps(error_state, indent=2), encoding="utf-8")
        _notify(progress_callback, "pipeline", "error", str(exc))
        return error_state
