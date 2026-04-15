"""FastAPI application entry point and routes.

Exposes the autonomous data-science pipeline as a REST API.
"""

import os
import io
import logging
import tempfile
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from auto_ds_agent.orchestrator.graph import run_pipeline
from auto_ds_agent.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="Auto DS Agent API",
    description="Autonomous Data Scientist Agent — upload a dataset and receive a full analysis.",
    version="1.0.0",
)

# CORS (allow all during development; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """Basic liveness probe."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@app.post("/analyze", tags=["pipeline"])
async def analyze_dataset(
    file: UploadFile = File(..., description="CSV or Excel dataset"),
    user_goal: Optional[str] = Form(default="", description="Natural-language analysis goal"),
    target_col: Optional[str] = Form(default="", description="Explicit target column name"),
) -> JSONResponse:
    """Upload a dataset and run the full autonomous analysis pipeline.

    Returns a JSON object with:
      - plan, cleaning_report, eda_insights, model_results,
        evaluation_report, final_report_md, best_model_name
    """
    # 1. Read the uploaded file into a DataFrame
    filename = file.filename or "upload"
    content = await file.read()

    try:
        if filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        logger.error("Failed to parse uploaded file '%s': %s", filename, exc)
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}")

    logger.info("Received dataset '%s' with shape %s", filename, df.shape)

    # 2. Save raw dataset
    raw_path = os.path.join(settings.datasets_dir, filename)
    with open(raw_path, "wb") as f:
        f.write(content)

    # 3. Run pipeline
    try:
        final_state = run_pipeline(
            df=df,
            user_goal=user_goal or "",
            target_col=target_col or "",
        )
    except Exception as exc:
        logger.exception("Pipeline execution failed.")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # 4. Build response (exclude non-serialisable objects)
    response = {
        "plan": final_state.get("plan"),
        "cleaning_report": final_state.get("cleaning_report"),
        "eda_insights": final_state.get("eda_insights"),
        "eda_plots": final_state.get("eda_plots"),
        "model_results": final_state.get("model_results"),
        "best_model_name": final_state.get("best_model_name"),
        "cv_results": final_state.get("cv_results"),
        "feature_importances": final_state.get("feature_importances"),
        "evaluation_report": final_state.get("evaluation_report"),
        "final_report_md": final_state.get("final_report_md"),
        "report_path": final_state.get("report_path"),
    }

    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# Report download
# ---------------------------------------------------------------------------

@app.get("/report", tags=["pipeline"])
async def download_report() -> FileResponse:
    """Download the latest generated Markdown report."""
    report_path = os.path.join(settings.outputs_dir, "final_report.md")
    if not os.path.isfile(report_path):
        raise HTTPException(status_code=404, detail="No report found. Run /analyze first.")
    return FileResponse(report_path, media_type="text/markdown", filename="final_report.md")
