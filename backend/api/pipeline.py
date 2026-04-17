# backend/api/pipeline.py
import json
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from agents.evaluator import EvaluatorAgent
from orchestrator.graph import run_pipeline
from utils import progress_store
from utils.serializer import make_serializable

router = APIRouter()


def _read_dataset(path: Path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise HTTPException(status_code=422, detail="File must be CSV or Excel.")


def _background_run(df, goal, target_col, selected_models, user_params, fast_mode, run_id, dataset_path, dataset_name, created_at, run_explainability):
    try:
        run_pipeline(
            df,
            goal,
            target_col,
            selected_models,
            user_params,
            fast_mode,
            run_id,
            progress_callback=lambda agent, status, message: progress_store.update(run_id, agent, status, message),
            dataset_path=str(dataset_path),
            dataset_name=dataset_name,
            created_at=created_at,
            run_explainability=run_explainability,
        )
    except Exception as exc:
        progress_store.update(run_id, "pipeline", "error", str(exc))


@router.post("/run")
async def start_pipeline(
    background_tasks: BackgroundTasks,
    goal: str = Form(...),
    target_col: str = Form(None),
    selected_models: str = Form("[]"),
    fast_mode: bool = Form(False),
    run_explainability: bool = Form(True),
    file: UploadFile = File(...),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=422, detail="File must be CSV or Excel.")
    run_id = str(uuid.uuid4())[:8]
    uploads_dir = Path("./storage/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "dataset.csv").name
    uploaded_path = uploads_dir / f"{run_id}_{safe_name}"
    uploaded_path.write_bytes(await file.read())
    df = _read_dataset(uploaded_path)
    if df.shape[0] < 50:
        raise HTTPException(status_code=422, detail="Dataset must have at least 50 rows.")
    if df.shape[1] < 2:
        raise HTTPException(status_code=422, detail="Dataset must have at least 2 columns.")
    if target_col and target_col not in df.columns:
        raise HTTPException(status_code=422, detail=f"Target column '{target_col}' does not exist in the dataset.")
    try:
        parsed_models = json.loads(selected_models) if selected_models else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="selected_models must be a JSON array.")
    created_at = time.time()
    progress_store.update(run_id, "planner", "running", "Starting pipeline...")
    background_tasks.add_task(_background_run, df, goal, target_col or None, parsed_models, {}, fast_mode, run_id, uploaded_path, safe_name, created_at, run_explainability)
    return {"run_id": run_id, "status": "started"}


@router.get("/status/{run_id}")
def pipeline_status(run_id: str):
    status = progress_store.get(run_id)
    run_path = Path(f"./storage/runs/{run_id}.json")
    if status.get("status") == "not_found" and run_path.exists():
        data = json.loads(run_path.read_text(encoding="utf-8"))
        stored_status = data.get("status", "completed")
        return {"status": stored_status, "progress_pct": 100 if stored_status == "completed" else 0, "current_agent": "pipeline", "completed_agents": [], "logs": [], "error": data.get("error")}
    return status


@router.get("/result/{run_id}")
def pipeline_result(run_id: str):
    path = Path(f"./storage/runs/{run_id}.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    ml = data.get("ml_results", {})
    evaluation = data.get("evaluation_results", {})
    if ml and evaluation.get("status") == "error":
        problem_type = ml.get("problem_type") or data.get("problem_type", "classification")
        data["evaluation_results"] = EvaluatorAgent().run(ml, ml.get("y_test", []), ml.get("y_pred", []), problem_type)
    return JSONResponse(content=make_serializable(data))
