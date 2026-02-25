"""Page 1: Data Collection — BYOD upload or Literature Mining Agent."""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import streamlit as st

from config import RAW_DIR, HEA_PROPERTIES
from core.state import WorkflowState
from core.llm_client import create_llm_client
from core.theme import inject_theme, page_hero


def _get_state() -> WorkflowState:
    return st.session_state.workflow_state


def _log(msg: str):
    st.session_state.agent_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


inject_theme()
page_hero("Data Collection", "Upload your HEA dataset or mine data from literature databases & materials repositories", "🔬")

state = _get_state()

# ── Mode selection ────────────────────────────────────────────────────────────
tab_byod, tab_lit = st.tabs(["📁 Upload Your Data (BYOD)", "📚 Collect from Literature"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: BYOD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_byod:
    st.markdown("""
    Upload a **CSV or Excel file** with at least:
    - A column containing HEA **compositions** (e.g., `Al0.5CoCrFeNi`)
    - One or more columns containing **measured property values**
    """)

    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls"],
        help="CSV or Excel file with composition and property columns",
    )

    if uploaded_file:
        try:
            import pandas as pd
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns")
            st.dataframe(df.head(10), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                composition_col = st.selectbox(
                    "Composition column",
                    options=df.columns.tolist(),
                    index=next((i for i, c in enumerate(df.columns) if "comp" in c.lower()), 0),
                )
            with col2:
                target_col = st.selectbox(
                    "Target property column",
                    options=[c for c in df.columns if c != composition_col],
                )

            if st.button("Use This Dataset", type="primary", use_container_width=True):
                # Save to raw dir
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = RAW_DIR / f"user_upload_{ts}.csv"
                df.to_csv(out_path, index=False)

                state.raw_data_path = str(out_path)
                state.n_raw_samples = len(df)
                state.target_property = target_col
                state.data_entry_mode = "byod"

                _log(f"Dataset uploaded: {len(df)} rows, target={target_col}")
                st.success(f"Dataset saved! {len(df)} samples ready for feature engineering.")
                st.info("Proceed to **Feature Engineering** in the sidebar.")

        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Show current dataset if already loaded
    if state.raw_data_path and Path(state.raw_data_path).exists():
        st.divider()
        st.success(f"Current dataset: `{Path(state.raw_data_path).name}` ({state.n_raw_samples} samples)")
        if state.target_property:
            st.info(f"Target property: **{state.target_property}**")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Literature Mining
# ═══════════════════════════════════════════════════════════════════════════════
with tab_lit:
    st.markdown("Use the **Literature Mining Agent** to collect HEA data from databases and papers.")

    col1, col2 = st.columns(2)
    with col1:
        target_property = st.selectbox(
            "Target Property",
            options=HEA_PROPERTIES,
            key="lit_target_property",
        )

        elements_input = st.text_input(
            "Elements of Interest (comma-separated)",
            value="Al, Co, Cr, Fe, Ni",
            help="Common HEA elements. E.g.: Al, Co, Cr, Fe, Ni",
        )
        elements = [e.strip() for e in elements_input.split(",") if e.strip()]

    with col2:
        sources = st.multiselect(
            "Data Sources",
            options=["materials_project", "aflow", "arxiv"],
            default=["materials_project", "arxiv"],
        )

        arxiv_query = st.text_input(
            "arXiv Search Query",
            value=f"high entropy alloy {target_property} machine learning",
        )
        max_papers = st.slider("Max arXiv papers to parse", 1, 30, 10)

    # API Key check
    import os
    mp_key = os.getenv("MP_API_KEY", "")
    if "materials_project" in sources and not mp_key:
        st.warning("Materials Project requires an API key. Set `MP_API_KEY` in your `.env` file.")

    if st.button("🚀 Run Literature Mining Agent", type="primary", use_container_width=True,
                 disabled=not sources):
        with st.spinner("Literature Mining Agent is working..."):
            log_container = st.empty()
            logs: list[str] = []

            def stream_log(agent_name: str, msg: str):
                logs.append(f"[{agent_name}] {msg}")
                log_container.code("\n".join(logs[-20:]))

            try:
                llm = create_llm_client(state.llm_provider, state.llm_model)
                from agents.supervisor import OrchestratorAgent
                orchestrator = OrchestratorAgent(llm, state, log_callback=stream_log)

                result = orchestrator.run_literature_workflow(
                    target_property=target_property,
                    elements=elements,
                    sources=sources,
                    arxiv_query=arxiv_query,
                )

                state.target_property = target_property
                state.elements = elements
                st.success("Literature mining complete!")
                st.markdown("**Agent Summary:**")
                st.markdown(result)
                _log(f"Literature mining done: {target_property}, {elements}")

            except Exception as e:
                st.error(f"Agent error: {e}")
                _log(f"ERROR: {e}")

    # Show status
    if state.raw_data_path:
        st.divider()
        st.success(f"Data collected: `{Path(state.raw_data_path).name}`")
        st.info("Proceed to **Feature Engineering** →")

# ── Agent log expander ────────────────────────────────────────────────────────
if st.session_state.agent_logs:
    with st.expander("Agent Log", expanded=False):
        st.code("\n".join(st.session_state.agent_logs[-50:]))
