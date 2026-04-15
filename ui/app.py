"""Streamlit application frontend.

Provides a drag-and-drop interface for uploading datasets,
configuring goals, and viewing the full analysis pipeline output.
"""

import os
import io
import logging

import pandas as pd
import streamlit as st

from auto_ds_agent.orchestrator.graph import run_pipeline
from auto_ds_agent.config.settings import get_settings

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
st.markdown("Upload a dataset, set an optional goal, and let the agent handle the rest.")

if uploaded_file is not None and run_btn:
    # 1. Parse file
    try:
        if uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Failed to read file: {exc}")
        st.stop()

    st.subheader("📋 Raw Dataset Preview")
    st.dataframe(df.head(20), width="stretch")
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # 2. Run pipeline
    with st.spinner("Running the full analysis pipeline… this may take a minute."):
        try:
            settings = get_settings()
            final_state = run_pipeline(
                df=df,
                user_goal=user_goal or "",
                target_col=target_col or "",
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            logger.exception("Pipeline failed in Streamlit.")
            st.stop()

    st.success("Pipeline complete!")

    # 3. Display results in tabs
    tabs = st.tabs([
        "📌 Plan", "🧹 Cleaning", "📊 EDA",
        "🤖 Models", "📈 Evaluation", "📝 Report",
    ])

    # --- Plan ---
    with tabs[0]:
        st.json(final_state.get("plan", {}))

    # --- Cleaning ---
    with tabs[1]:
        st.json(final_state.get("cleaning_report", {}))

    # --- EDA ---
    with tabs[2]:
        eda_insights = final_state.get("eda_insights", {})
        st.json(eda_insights)

        eda_plots = final_state.get("eda_plots", {})
        if eda_plots:
            st.subheader("EDA Plots")
            cols = st.columns(2)
            for idx, (name, path) in enumerate(eda_plots.items()):
                if path and os.path.isfile(path):
                    with cols[idx % 2]:
                        st.image(path, caption=name, use_container_width=True)

    # --- Models ---
    with tabs[3]:
        st.json(final_state.get("model_results", {}))
        best = final_state.get("best_model_name", "N/A")
        st.info(f"**Best model:** {best}")

    # --- Evaluation ---
    with tabs[4]:
        st.json(final_state.get("evaluation_report", {}))
        cv = final_state.get("cv_results", {})
        if cv:
            st.subheader("Cross-Validation")
            st.json(cv)
        fi = final_state.get("feature_importances", {})
        if fi:
            st.subheader("Feature Importances")
            st.json(fi)

    # --- Report ---
    with tabs[5]:
        md = final_state.get("final_report_md", "")
        if md:
            st.markdown(md)
            st.download_button(
                "⬇️ Download Report",
                data=md,
                file_name="final_report.md",
                mime="text/markdown",
            )
        else:
            st.warning("No report generated.")

elif uploaded_file is None:
    st.info("👈 Upload a dataset from the sidebar to get started.")
