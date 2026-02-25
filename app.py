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
from core.theme import inject_theme

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HEAMLAgentic",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

# ── Session state initialization ──────────────────────────────────────────────
if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = WorkflowState()

if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="text-align:center; padding: 0.5rem 0 1rem;">
    <div style="font-size:2.4rem; margin-bottom:0.2rem;">⚗️</div>
    <div style="font-size:1.1rem; font-weight:700; background:linear-gradient(135deg,#1e90ff,#00d4aa);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">HEAMLAgentic</div>
    <div style="font-size:0.72rem; color:#7a9cc0; letter-spacing:0.08em; text-transform:uppercase; margin-top:2px;">
        High Entropy Alloy · ML Discovery
    </div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # LLM Provider selection
    st.markdown("**🤖 LLM Provider**")
    provider = st.selectbox(
        "Provider",
        options=list(PROVIDER_MODELS.keys()),
        index=list(PROVIDER_MODELS.keys()).index(
            st.session_state.workflow_state.llm_provider
        ),
        key="provider_select",
        label_visibility="collapsed",
    )
    model = st.selectbox(
        "Model",
        options=PROVIDER_MODELS[provider],
        index=0,
        key="model_select",
        label_visibility="collapsed",
    )

    # Update state when provider/model changes
    if (provider != st.session_state.workflow_state.llm_provider or
            model != st.session_state.workflow_state.llm_model):
        st.session_state.workflow_state.llm_provider = provider
        st.session_state.workflow_state.llm_model = model

    st.divider()

    # Workflow status metrics
    st.markdown("**📊 Workflow Status**")
    state: WorkflowState = st.session_state.workflow_state

    def _status_icon(value) -> str:
        return "✅" if value else "⬜"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Samples", state.n_raw_samples or state.n_samples or "—")
        st.metric("Features", state.n_features or "—")
    with col2:
        st.metric("Model", state.best_model_type or "—")
        st.metric(
            "CV R²",
            f"{state.best_cv_r2:.3f}" if state.best_cv_r2 is not None else "—"
        )

    st.divider()

    # Steps checklist
    st.markdown("**🔬 Pipeline Steps**")
    steps = [
        (state.raw_data_path or state.n_raw_samples, "Data Ready"),
        (state.processed_data_path, "Features Engineered"),
        (state.best_model_path, "Model Trained"),
        (state.report_path, "Report Generated"),
    ]
    for i, (done, label) in enumerate(steps, 1):
        icon = "✅" if done else "⬜"
        color = "#00d4aa" if done else "#7a9cc0"
        st.markdown(
            f'<div class="step-item"><span>{icon}</span>'
            f'<span style="color:{color}; font-size:0.83rem;">{i}. {label}</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Reset button
    if st.button("↺  Reset Workflow", type="secondary", use_container_width=True):
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
