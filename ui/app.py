"""Streamlit application frontend.

Provides a drag-and-drop interface for uploading datasets,
configuring goals, and viewing the full analysis pipeline output.
"""

import os
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Fix import path (run from project root)
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from orchestrator.graph import run_pipeline
from config.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Auto DS Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🤖 Auto DS Agent")
st.sidebar.markdown("**Autonomous Data Scientist**")

uploaded_file = st.sidebar.file_uploader(
    "Upload a dataset (CSV / Excel)", type=["csv", "xlsx", "xls"]
)

user_goal = st.sidebar.text_area(
    "Analysis goal (optional)",
    placeholder="e.g. Predict customer churn using classification",
)

target_col = st.sidebar.text_input(
    "Target column (optional)",
    placeholder="e.g. target, label, churn",
)

run_btn = st.sidebar.button("🚀 Run Pipeline", use_container_width=True)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🤖 Autonomous Data Scientist Agent")
st.markdown(
    "Upload a dataset, set an optional goal, and let the agent handle the rest."
)

if uploaded_file is not None and run_btn:

    # -----------------------------------------------------------------------
    # Parse uploaded file
    # -----------------------------------------------------------------------

    try:
        if uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

    except Exception as exc:
        st.error(f"Failed to read file: {exc}")
        st.stop()

    st.subheader("📋 Raw Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # -----------------------------------------------------------------------
    # Run pipeline
    # -----------------------------------------------------------------------

    with st.spinner("Running full analysis pipeline... Please wait."):

        try:
            settings = get_settings()

            final_state = run_pipeline(
                df=df,
                user_goal=user_goal.strip(),
                target_col=target_col.strip(),
            )

        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            logger.exception("Pipeline execution failed.")
            st.stop()

    st.success("✅ Pipeline completed successfully!")

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------

    tabs = st.tabs(
        [
            "📌 Plan",
            "🧹 Cleaning",
            "📊 EDA",
            "🤖 Models",
            "📈 Evaluation",
            "📝 Report",
        ]
    )

    # -----------------------------------------------------------------------
    # Plan
    # -----------------------------------------------------------------------

    with tabs[0]:
        st.json(final_state.get("plan", {}))

    # -----------------------------------------------------------------------
    # Cleaning
    # -----------------------------------------------------------------------

    with tabs[1]:
        st.json(final_state.get("cleaning_report", {}))

    # -----------------------------------------------------------------------
    # EDA
    # -----------------------------------------------------------------------

    with tabs[2]:
        st.json(final_state.get("eda_insights", {}))

        eda_plots = final_state.get("eda_plots", {})
        if eda_plots:
            st.subheader("📊 Visualizations")

            cols = st.columns(2)

            for idx, (name, path) in enumerate(eda_plots.items()):
                if path and os.path.exists(path):
                    with cols[idx % 2]:
                        st.image(path, caption=name, use_container_width=True)

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------

    with tabs[3]:
        st.json(final_state.get("model_results", {}))
        st.info(f"🏆 Best Model: {final_state.get('best_model_name', 'N/A')}")

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    with tabs[4]:
        st.json(final_state.get("evaluation_report", {}))

        cv = final_state.get("cv_results", {})
        if cv:
            st.subheader("Cross Validation")
            st.json(cv)

        fi = final_state.get("feature_importances", {})
        if fi:
            st.subheader("Feature Importance")
            st.json(fi)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------

    with tabs[5]:
        md = final_state.get("final_report_md", "")

        if md:
            st.markdown(md)

            st.download_button(
                label="⬇️ Download Report",
                data=md,
                file_name="final_report.md",
                mime="text/markdown",
            )
        else:
            st.warning("No report generated.")

# ---------------------------------------------------------------------------
# Initial Screen
# ---------------------------------------------------------------------------

else:
    st.info("👈 Upload a dataset from the sidebar and click Run Pipeline.")