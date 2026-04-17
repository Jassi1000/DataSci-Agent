# backend/api/export.py
import json
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from utils.report_exporter import export_summary_csv, export_to_excel, export_to_pdf

router = APIRouter()


def _load(run_id: str):
    path = Path(f"./storage/runs/{run_id}.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/pdf/{run_id}")
def pdf(run_id: str):
    pdf_bytes = export_to_pdf(_load(run_id), run_id)
    return StreamingResponse(BytesIO(pdf_bytes), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=insightforge_{run_id}.pdf"})


@router.get("/excel/{run_id}")
def excel(run_id: str):
    excel_bytes = export_to_excel(_load(run_id), run_id)
    return StreamingResponse(BytesIO(excel_bytes), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename=insightforge_{run_id}.xlsx"})


@router.get("/csv/{run_id}")
def csv(run_id: str):
    csv_str = export_summary_csv(_load(run_id))
    return StreamingResponse(iter([csv_str]), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=insightforge_{run_id}.csv"})

