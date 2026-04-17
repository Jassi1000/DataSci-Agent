# backend/api/history.py
import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def history():
    runs_dir = Path("./storage/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for f in runs_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append(
                {
                    "run_id": f.stem,
                    "dataset_name": data.get("dataset_name", f.stem),
                    "created_at": data.get("created_at", f.stat().st_mtime),
                    "status": data.get("status", "completed"),
                    "best_model": data.get("ml_results", {}).get("best_model_name", "N/A"),
                    "score": data.get("ml_results", {}).get("best_metrics", {}).get("f1_weighted") or data.get("ml_results", {}).get("best_metrics", {}).get("r2"),
                    "problem_type": data.get("ml_results", {}).get("problem_type") or data.get("eda_results", {}).get("problem_type", "unknown"),
                    "goal": data.get("goal", ""),
                }
            )
        except Exception:
            continue
    return sorted(results, key=lambda x: x["created_at"], reverse=True)


@router.delete("/{run_id}")
def delete_history(run_id: str):
    path = Path(f"./storage/runs/{run_id}.json")
    if path.exists():
        path.unlink()
    return {"deleted": True}

