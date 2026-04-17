"""Application settings and environment variable management.

Uses python-dotenv to load .env into os.environ so that ALL
consumers (agents, tools, libraries) can read env vars directly.
Then pydantic-settings provides typed, validated access.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate and load .env into os.environ BEFORE anything else.
# Resolves relative to the auto_ds_agent package root so it works
# regardless of the working directory.
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # auto_ds_agent/
_ENV_FILE = _PACKAGE_ROOT / ".env"

# override=False keeps explicit env vars from being clobbered
load_dotenv(dotenv_path=_ENV_FILE, override=False)


class Settings(BaseSettings):
    """Central configuration for the Auto DS Agent application."""

    # -- LLM -----------------------------------------------------------------
    groq_api_key: str = Field(default="", description="Groq API key")
    llm_model_name: str = Field(default="llama-3.3-70b-versatile", description="Groq model name")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM sampling temperature")

    # -- Data -----------------------------------------------------------------
    datasets_dir: str = Field(default="storage/datasets", description="Directory for uploaded datasets")
    outputs_dir: str = Field(default="storage/outputs", description="Directory for generated outputs")
    logs_dir: str = Field(default="storage/logs", description="Directory for log files")
    saved_models_dir: str = Field(default="models/saved_models", description="Directory for persisted models")

    # -- ML -------------------------------------------------------------------
    test_size: float = Field(default=0.2, ge=0.05, le=0.5, description="Train/test split ratio")
    cv_folds: int = Field(default=5, ge=2, le=20, description="Number of cross-validation folds")
    cardinality_threshold: int = Field(
        default=10, ge=2,
        description="Max unique values for one-hot encoding (above → label encode)",
    )

    # -- API ------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="FastAPI port")

    # -- Streamlit ------------------------------------------------------------
    streamlit_port: int = Field(default=8501, ge=1, le=65535, description="Streamlit port")

    # -- General --------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = {
        "env_file": str(_ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Instantiate and return the application settings singleton.

    Ensures required directories exist on first access.
    """
    settings = Settings()

    # Create storage directories
    for dir_path in (settings.datasets_dir, settings.outputs_dir, settings.logs_dir, settings.saved_models_dir):
        os.makedirs(dir_path, exist_ok=True)

    # Set log level globally
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Settings loaded. LLM model: %s", settings.llm_model_name)
    return settings
