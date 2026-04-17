# backend/tools/ml_tools.py


def primary_metric(problem_type: str) -> str:
    return "f1_weighted" if problem_type == "classification" else "r2"


def format_score(value) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"

