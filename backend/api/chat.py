# backend/api/chat.py
import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.chat_agent import ChatAgent

router = APIRouter()
_agents: dict = {}


class ChatRequest(BaseModel):
    question: str
    history: list = Field(default_factory=list)


@router.post("/{run_id}")
def chat(run_id: str, request: ChatRequest):
    try:
        run_path = Path(f"./storage/runs/{run_id}.json")
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found.")
        pipeline_data = json.loads(run_path.read_text(encoding="utf-8"))
        dataset_path = Path(pipeline_data["dataset_path"])
        df = pd.read_csv(dataset_path) if dataset_path.suffix.lower() == ".csv" else pd.read_excel(dataset_path)
        if run_id not in _agents:
            agent = ChatAgent()
            agent.build_index(df, pipeline_data)
            _agents[run_id] = agent
        answer = _agents[run_id].answer(request.question, request.history, pipeline_data)
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

