"""HEAMLAgentic — Streamlit entry point."""

from __future__ import annotations
import os
from pathlib import Path

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import streamlit as st

from core.state import WorkflowState
from core.llm_client import PROVIDER_MODELS

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HEAMLAgentic",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialization ──────────────────────────────────────────────
if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = WorkflowState()

if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚗️ HEAMLAgentic")
    st.caption("High Entropy Alloy ML Discovery")

    st.divider()

    # LLM Provider selection
    st.subheader("LLM Provider")
    provider = st.selectbox(
        "Provider",
        options=list(PROVIDER_MODELS.keys()),
        index=list(PROVIDER_MODELS.keys()).index(
            st.session_state.workflow_state.llm_provider
        ),
        key="provider_select",
    )
    model = st.selectbox(
        "Model",
        options=PROVIDER_MODELS[provider],
        index=0,
        key="model_select",
    )

    # Update state when provider/model changes
    if (provider != st.session_state.workflow_state.llm_provider or
            model != st.session_state.workflow_state.llm_model):
        st.session_state.workflow_state.llm_provider = provider
        st.session_state.workflow_state.llm_model = model

    st.divider()

    # Workflow status
    st.subheader("Workflow Status")
    state: WorkflowState = st.session_state.workflow_state

    def _status_icon(value) -> str:
        return "✅" if value else "⬜"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Data Points", state.n_raw_samples or state.n_samples or "—")
        st.metric("Features", state.n_features or "—")
    with col2:
        st.metric("Best Model", state.best_model_type or "—")
        st.metric(
            "Best R²",
            f"{state.best_cv_r2:.3f}" if state.best_cv_r2 is not None else "—"
        )

    st.divider()

    # Steps checklist
    st.subheader("Steps")
    st.write(f"{_status_icon(state.raw_data_path or state.n_raw_samples)} 1. Data Ready")
    st.write(f"{_status_icon(state.processed_data_path)} 2. Features Engineered")
    st.write(f"{_status_icon(state.best_model_path)} 3. Model Trained")
    st.write(f"{_status_icon(state.report_path)} 4. Report Generated")

    st.divider()

    # Reset button
    if st.button("Reset Workflow", type="secondary", use_container_width=True):
        st.session_state.workflow_state.reset()
        st.session_state.agent_logs = []
        st.rerun()

# ── Navigation ────────────────────────────────────────────────────────────────
pages = [
    st.Page("pages/01_data_collection.py", title="1. Data Collection", icon="🔬"),
    st.Page("pages/02_data_processing.py", title="2. Feature Engineering", icon="⚙️"),
    st.Page("pages/03_automl.py", title="3. AutoML", icon="🤖"),
    st.Page("pages/04_evaluation.py", title="4. Evaluation", icon="📊"),
    st.Page("pages/05_predict.py", title="5. Predict", icon="🔮"),
]

pg = st.navigation(pages)
pg.run()
