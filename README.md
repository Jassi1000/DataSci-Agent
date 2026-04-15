# 🤖 Auto DS Agent — Autonomous Data Scientist

A production-grade, multi-agent system that autonomously analyses datasets end-to-end:
data cleaning → EDA → model training → evaluation → report generation.

## Architecture

```
User uploads CSV/Excel
        │
        ▼
┌──────────────┐
│  PlannerAgent │  ← LLM generates execution plan
└──────┬───────┘
       ▼
┌──────────────┐
│   DataAgent  │  ← Cleans, imputes, encodes
└──────┬───────┘
       ▼
┌──────────────┐
│   EDAAgent   │  ← Stats, plots, LLM insights
└──────┬───────┘
       ▼
┌──────────────┐
│    MLAgent   │  ← Trains multiple models
└──────┬───────┘
       ▼
┌──────────────┐
│  Evaluator   │  ← Cross-val, feature importance, verdicts
└──────┬───────┘
       ▼
┌──────────────┐
│   Reporter   │  ← Compiles Markdown report
└──────────────┘
```

## Tech Stack

| Layer          | Technology                        |
|----------------|-----------------------------------|
| LLM            | Groq (LLaMA 3 70B) via LangChain |
| Orchestration  | LangGraph                         |
| ML             | scikit-learn                      |
| Data           | Pandas, NumPy                     |
| Visualisation  | Matplotlib, Seaborn               |
| API            | FastAPI                           |
| UI             | Streamlit                         |
| Validation     | Pydantic                          |

## Quick Start

```bash
# 1. Clone and enter the project
cd auto_ds_agent

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key
#    Edit .env and replace your_groq_api_key_here with your actual key

# 5a. Run the Streamlit UI
streamlit run ui/app.py

# 5b. Or run the FastAPI backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
auto_ds_agent/
├── agents/              # All autonomous agents
│   ├── planner.py       # Task decomposition & plan generation
│   ├── data_agent.py    # Data cleaning & preprocessing
│   ├── eda_agent.py     # Exploratory data analysis
│   ├── ml_agent.py      # Model selection & training
│   ├── evaluator.py     # Model evaluation & verdicts
│   └── reporter.py      # Final report compilation
├── orchestrator/
│   └── graph.py         # LangGraph DAG pipeline
├── tools/               # Pure utility functions (no LLM)
│   ├── data_tools.py    # Pandas/NumPy operations
│   ├── ml_tools.py      # scikit-learn training & metrics
│   └── viz_tools.py     # Matplotlib/Seaborn plotting
├── config/
│   ├── settings.py      # Pydantic-settings configuration
│   └── prompts.py       # Central prompt registry
├── api/
│   └── main.py          # FastAPI REST endpoints
├── ui/
│   └── app.py           # Streamlit frontend
├── storage/
│   ├── datasets/        # Uploaded datasets
│   ├── outputs/         # Generated plots & reports
│   └── logs/            # Application logs
├── models/
│   └── saved_models/    # Persisted sklearn models (.pkl)
├── tests/               # Test suite
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## API Endpoints

| Method | Path       | Description                    |
|--------|------------|--------------------------------|
| GET    | `/health`  | Liveness probe                 |
| POST   | `/analyze` | Upload dataset & run pipeline  |
| GET    | `/report`  | Download latest Markdown report|

## Running Tests

```bash
pytest tests/ -v
```

## Docker (future)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## License

MIT
