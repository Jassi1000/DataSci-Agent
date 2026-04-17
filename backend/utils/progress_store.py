# backend/utils/progress_store.py
import threading
import time

_store = {}
_lock = threading.Lock()


def update(run_id: str, agent_name: str, status: str, message: str = ""):
    """Update progress. Status values: running, done, error, completed."""
    with _lock:
        if run_id not in _store:
            _store[run_id] = {
                "status": "running",
                "current_agent": agent_name,
                "completed_agents": [],
                "logs": [],
                "started_at": time.time(),
                "total_agents": 7,
                "progress_pct": 0,
            }
        entry = _store[run_id]
        entry["current_agent"] = agent_name
        entry["logs"].append(
            {
                "agent": agent_name,
                "status": status,
                "message": message,
                "timestamp": time.strftime("%H:%M:%S"),
            }
        )
        if status == "done" and agent_name not in entry["completed_agents"]:
            entry["completed_agents"].append(agent_name)
        entry["progress_pct"] = int(
            len(entry["completed_agents"]) / entry["total_agents"] * 100
        )
        if status == "completed":
            entry["status"] = "completed"
            entry["progress_pct"] = 100
        elif status == "error":
            entry["status"] = "failed"
            entry["error"] = message
        elif entry.get("status") != "failed":
            entry["status"] = "running"


def get(run_id: str) -> dict:
    with _lock:
        store_data = _store.get(run_id, {"status": "not_found"})
        result = dict(store_data)
        if "started_at" in store_data:
            result["elapsed_seconds"] = int(time.time() - store_data["started_at"])
        return result

