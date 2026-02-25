"""Page 2: Feature Engineering — DataProcessingAgent."""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import streamlit as st

from config import FEATURIZER_SETS
from core.state import WorkflowState
from core.llm_client import create_llm_client
from core.theme import inject_theme, page_hero


def _get_state() -> WorkflowState:
    return st.session_state.workflow_state


def _log(msg: str):
    st.session_state.agent_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


inject_theme()
page_hero("Feature Engineering", "Transform HEA compositions into 142 physics-informed features using matminer (Magpie, Yang, ValenceOrbital)", "⚙️")

state = _get_state()

# ── Prerequisites check ───────────────────────────────────────────────────────
if not state.raw_data_path:
    st.warning("No dataset loaded yet. Go to **Data Collection** first.")
    st.stop()

st.success(f"Input data: `{Path(state.raw_data_path).name}` ({state.n_raw_samples or '?'} samples)")

# ── Configuration ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    target_col = st.text_input(
        "Target Property Column",
        value=state.target_property or "",
        help="Column name in your dataset that contains the property values to predict",
    )
    composition_col = st.text_input(
        "Composition Column",
        value="composition",
        help="Column containing HEA composition strings",
    )

with col2:
    featurizer_set = st.radio(
        "Featurizer Set",
        options=["minimal", "standard", "comprehensive"],
        index=1,
        help="minimal=~20 features (fast), standard=~80 features (recommended), comprehensive=~200 features",
    )

    iqr_factor = st.slider(
        "Outlier IQR Factor",
        min_value=1.5,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="Outliers beyond IQR×factor are removed from target column",
    )

# ── Run button ────────────────────────────────────────────────────────────────
if not target_col:
    st.error("Please specify the target property column name.")
    st.stop()

if st.button("⚙️ Run Data Processing Agent", type="primary", use_container_width=True):
    log_container = st.empty()
    logs: list[str] = []

    def stream_log(agent_name: str, msg: str):
        logs.append(f"[{agent_name}] {msg}")
        log_container.code("\n".join(logs[-25:]))

    with st.spinner("Processing dataset and engineering features..."):
        try:
            llm = create_llm_client(state.llm_provider, state.llm_model)
            from agents.data_agent import DataProcessingAgent
            agent = DataProcessingAgent(llm, state)
            agent.set_log_callback(stream_log)

            result = agent.process(
                input_path=state.raw_data_path,
                target_column=target_col,
                featurizer_set=featurizer_set,
                composition_column=composition_col,
            )

            state.target_property = target_col
            state.featurizer_set = featurizer_set
            _log(f"Feature engineering done: {featurizer_set}")

            st.success("Feature engineering complete!")
            st.markdown("**Agent Summary:**")
            st.markdown(result)

            # Try to parse and display stats from result
            try:
                import json, re
                # Find the featurized path in the result text
                if "featurized_path" in result or "processed" in result:
                    import pandas as pd
                    from pathlib import Path
                    from config import PROCESSED_DIR

                    # Find the most recent features parquet
                    parquet_files = sorted(PROCESSED_DIR.glob(f"features_{target_col}_*.parquet"))
                    if parquet_files:
                        latest = parquet_files[-1]
                        df = pd.read_parquet(latest)
                        state.processed_data_path = str(latest)
                        state.n_samples = len(df)
                        state.n_features = len([c for c in df.columns
                                                if c not in ("composition", target_col)])
                        feature_cols = [c for c in df.columns if c not in ("composition", target_col)]
                        state.feature_names = feature_cols

                        st.divider()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Samples", state.n_samples)
                        col2.metric("Features", state.n_features)
                        col3.metric("Target", target_col)

                        # Distribution plot
                        try:
                            import plotly.express as px
                            fig = px.histogram(df, x=target_col, nbins=30,
                                               title=f"{target_col} Distribution",
                                               labels={target_col: target_col})
                            st.plotly_chart(fig, use_container_width=True)
                        except ImportError:
                            pass
            except Exception:
                pass  # Stats display is best-effort

        except Exception as e:
            st.error(f"Agent error: {e}")
            _log(f"ERROR: {e}")

# ── Show current state ────────────────────────────────────────────────────────
if state.processed_data_path:
    st.divider()
    st.success(f"Featurized data: `{Path(state.processed_data_path).name}`")
    st.info(f"{state.n_samples} samples × {state.n_features} features → Ready for **AutoML** →")

# ── Agent log ─────────────────────────────────────────────────────────────────
if st.session_state.agent_logs:
    with st.expander("Agent Log", expanded=False):
        st.code("\n".join(st.session_state.agent_logs[-50:]))
