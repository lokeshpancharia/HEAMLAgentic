"""Page 3: AutoML — model search and Optuna HPO."""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import streamlit as st

from config import AUTOML_MODELS, AUTOML_DEFAULT_TRIALS, AUTOML_DEFAULT_CV_FOLDS
from core.state import WorkflowState
from core.llm_client import create_llm_client
from core.theme import inject_theme, page_hero


def _get_state() -> WorkflowState:
    return st.session_state.workflow_state


def _log(msg: str):
    st.session_state.agent_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


inject_theme()
page_hero("AutoML", "Automated model selection across RF · XGBoost · LightGBM · GBM with Optuna hyperparameter optimization", "🤖")

state = _get_state()

# ── Prerequisites ─────────────────────────────────────────────────────────────
if not state.processed_data_path:
    st.warning("No featurized data available. Complete **Feature Engineering** first.")
    st.stop()

st.success(
    f"Featurized data: `{Path(state.processed_data_path).name}` "
    f"({state.n_samples} samples × {state.n_features} features)"
)

# ── Configuration ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    task_type = st.radio("Task Type", ["regression", "classification"], index=0)
    cv_folds = st.slider("CV Folds", 3, 10, AUTOML_DEFAULT_CV_FOLDS)
    n_trials = st.slider(
        "HPO Trials per Model",
        10, 200, AUTOML_DEFAULT_TRIALS,
        help="More trials = better results but slower. 50 is a good balance.",
    )

with col2:
    selected_models = st.multiselect(
        "Models to Search",
        options=AUTOML_MODELS,
        default=["RandomForest", "XGBoost", "LightGBM"],
        help="Select which model types to include in the search",
    )

# ── Run AutoML ────────────────────────────────────────────────────────────────
if not selected_models:
    st.error("Select at least one model to search.")
    st.stop()

if st.button("🚀 Run AutoML Agent", type="primary", use_container_width=True):
    log_container = st.empty()
    logs: list[str] = []

    def stream_log(agent_name: str, msg: str):
        logs.append(f"[{agent_name}] {msg}")
        log_container.code("\n".join(logs[-25:]))

    with st.spinner(f"Searching {len(selected_models)} models with {n_trials} HPO trials each..."):
        try:
            llm = create_llm_client(state.llm_provider, state.llm_model)
            from agents.automl_agent import AutoMLAgent
            agent = AutoMLAgent(llm, state)
            agent.set_log_callback(stream_log)

            result = agent.find_best_model(
                data_path=state.processed_data_path,
                target_column=state.target_property,
                n_trials=n_trials,
                cv_folds=cv_folds,
                task_type=task_type,
                models=selected_models,
            )

            _log(f"AutoML done: {state.best_model_type or 'unknown'}")
            st.success("AutoML complete!")
            st.markdown("**Agent Summary:**")
            st.markdown(result)

            # Parse results from models dir
            try:
                import json
                from config import MODELS_DIR
                model_files = sorted(MODELS_DIR.glob(f"best_model_{state.target_property}_*.joblib"))
                if model_files:
                    latest_model = model_files[-1]
                    state.best_model_path = str(latest_model)
                    meta_path = latest_model.with_suffix(".json")
                    if meta_path.exists():
                        with open(meta_path) as f:
                            meta = json.load(f)
                        state.best_model_type = meta.get("model_type")
                        state.best_cv_r2 = meta.get("cv_r2_mean")
                        state.best_cv_rmse = meta.get("cv_rmse_mean")
                        state.best_params = meta.get("best_params")
                        state.model_metadata_path = str(meta_path)

                        st.divider()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Best Model", state.best_model_type)
                        c2.metric("CV R²", f"{state.best_cv_r2:.3f}")
                        c3.metric("CV RMSE", f"{state.best_cv_rmse:.2f}")
                        c4.metric("Samples", state.n_samples)

                        with st.expander("Best Hyperparameters"):
                            st.json(state.best_params)

            except Exception:
                pass  # Best-effort

        except Exception as e:
            st.error(f"Agent error: {e}")
            _log(f"ERROR: {e}")

# ── Current state ─────────────────────────────────────────────────────────────
if state.best_model_path:
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", state.best_model_type or "—")
    c2.metric("CV R²", f"{state.best_cv_r2:.3f}" if state.best_cv_r2 else "—")
    c3.metric("CV RMSE", f"{state.best_cv_rmse:.2f}" if state.best_cv_rmse else "—")
    st.info("Model trained! Proceed to **Evaluation** →")

# ── Agent log ─────────────────────────────────────────────────────────────────
if st.session_state.agent_logs:
    with st.expander("Agent Log", expanded=False):
        st.code("\n".join(st.session_state.agent_logs[-50:]))
