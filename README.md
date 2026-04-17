# InsightForge

**Turn Raw Data Into Decisions. Instantly.**

InsightForge is an AI-powered autonomous data science platform that transforms raw datasets into actionable insights through intelligent data cleaning, exploratory analysis, model training, and explainability—all without manual intervention.

---

## ✨ Features

- **Smart Data Cleaning** — Automatically detect and fix missing values, duplicates, and type inconsistencies
- **Exploratory Data Analysis** — Generate comprehensive statistical summaries and visualizations
- **Model Comparison** — Train and benchmark multiple ML algorithms with automatic hyperparameter tuning
- **Explainability** — Understand model decisions with SHAP-based feature importance analysis
- **Interactive Chat** — Ask natural language questions about your analysis results
- **Report Generation** — Export detailed findings in professional formats
- **Run History** — Track all analyses with versioning and reproducibility
- **Fast Mode** — Execute quick analyses for rapid iteration

---

## 🛠 Tech Stack

### Frontend
- **React 18** — Component-based UI framework
- **Vite** — Lightning-fast build tool
- **Tailwind CSS** — Utility-first styling
- **shadcn/ui** — High-quality component library
- **Recharts** — Data visualization library

### Backend
- **FastAPI** — High-performance async web framework
- **Python 3.10+** — Core language
- **scikit-learn** — Classical ML algorithms
- **XGBoost, LightGBM, CatBoost** — Gradient boosted trees
- **SHAP** — Explainability and interpretability
- **Groq API** — Fast LLM inference for natural language chat
- **pandas, NumPy, SciPy** — Data manipulation and scientific computing

### Storage & Infrastructure
- **Local filesystem** — Dataset and model persistence
- **JSON** — Lightweight data serialization
- **In-memory caching** — Fast result retrieval

---

## 📁 Project Structure

```
InsightForge/
├── frontend/                 # React + Vite web application
│   ├── src/
│   │   ├── components/      # Reusable React components
│   │   ├── pages/           # Page components
│   │   ├── api/             # API client functions
│   │   └── store/           # State management
│   └── package.json
├── backend/                  # FastAPI Python backend
│   ├── agents/              # AI agents (planner, data, EDA, ML, etc.)
│   ├── api/                 # API endpoints
│   ├── tools/               # Utility functions
│   ├── orchestrator/        # Workflow orchestration
│   ├── storage/             # Data persistence
│   └── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Node.js 16 or higher
- Git

### 1. Clone & Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (choose your OS):
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install advanced features
pip install lightgbm catboost shap faiss-cpu sentence-transformers imbalanced-learn kaleido
```

### 2. Configure Environment

```bash
# Create .env file
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Required: Groq API key for chat functionality
GROQ_API_KEY=gsk-your-groq-key-here

# LLM Configuration
LLM_MODEL_NAME=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.0

# Optional: Fast mode skips expensive computations
FAST_MODE=false

# Data storage directory
STORAGE_PATH=./storage
```

Get your Groq API key at: https://console.groq.com

### 3. Run Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Backend API will be available at: **http://localhost:8000**

API docs (Swagger UI): **http://localhost:8000/docs**

### 4. Setup & Run Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at: **http://localhost:5173**

---

## 📊 Usage

1. **Upload Dataset** — CSV, Excel, or Parquet formats
2. **Configure Analysis**:
   - Select target column (auto-detected)
   - Define analysis goal
   - Choose ML algorithms to compare
3. **Run Pipeline** — Automated workflow executes all stages
4. **Explore Results**:
   - View data cleaning reports
   - Analyze EDA visualizations
   - Compare model performance
   - Read feature importance
5. **Chat** — Ask questions about your analysis
6. **Export** — Download reports and trained models

---

## 🔧 API Endpoints

### Pipeline Management
- `POST /api/pipeline/run` — Start analysis pipeline
- `GET /api/pipeline/status/{run_id}` — Check execution status
- `GET /api/pipeline/result/{run_id}` — Retrieve results

### Chat
- `POST /api/chat/message` — Send question about analysis

### History
- `GET /api/history` — List all past runs
- `GET /api/history/{run_id}` — Get specific run details

### Reports
- `GET /api/export/report/{run_id}` — Export analysis report

Full API documentation available at `/docs` when backend is running.

---

## 📈 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | Required | API key for Groq LLM service |
| `LLM_MODEL_NAME` | llama-3.3-70b-versatile | Groq model identifier |
| `LLM_TEMPERATURE` | 0.0 | LLM creativity (0=deterministic, 1=creative) |
| `FAST_MODE` | false | Skip heavy computations (SHAP, visualization) |
| `STORAGE_PATH` | ./storage | Where to save datasets and models |

### Runtime Options

```bash
# Development mode with auto-reload
uvicorn main:app --reload --port 8000

# Production mode (no auto-reload)
uvicorn main:app --host 0.0.0.0 --port 8000

# With custom workers
uvicorn main:app --workers 4 --port 8000
```

---

## 🎯 Supported ML Algorithms

### Classification
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- Gradient Boosting Classifier
- SVM Classifier

### Regression
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor
- Ridge Regression
- SVR

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.10+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Frontend won't load
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 16+
```

### Groq API errors
- Verify `GROQ_API_KEY` is set correctly in `.env`
- Check API key hasn't expired (revoke and regenerate if needed)
- Ensure network connectivity to `api.groq.com`

### Out of memory errors
- Enable `FAST_MODE=true` to skip SHAP and visualizations
- Reduce dataset size or use `--workers 1` for uvicorn

---

## 🚀 Deployment

### Docker

```dockerfile
# Build image
docker build -t insightforge .

# Run container
docker run -p 8000:8000 -e GROQ_API_KEY=your_key insightforge
```

### Heroku / Cloud Platforms

```bash
# Set environment variables
heroku config:set GROQ_API_KEY=your_key

# Deploy
git push heroku main
```

---

## 🔮 Roadmap & Future Improvements

- [ ] Advanced time-series analysis
- [ ] Automated feature engineering
- [ ] Deep learning model support
- [ ] Multi-model ensemble strategies
- [ ] Drift detection & monitoring
- [ ] A/B testing utilities
- [ ] Cloud data source integrations (Snowflake, BigQuery, S3)
- [ ] Custom model uploads
- [ ] Real-time streaming data support
- [ ] Mobile app

---

## 📝 License

MIT License — See LICENSE file for details

---

## 👨‍💼 Author

Built with ❤️ by the Netsmartz Data Science team

- GitHub: [@netsmartz](https://github.com)
- Website: [www.netsmartz.com](https://www.netsmartz.com)
- Support: support@netsmartz.com

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support & Community

- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Share ideas in GitHub Discussions
- **Documentation**: Full docs at [docs.insightforge.dev](https://docs.insightforge.dev)
- **Email**: support@netsmartz.com

---

**Happy analyzing! 🎉**
