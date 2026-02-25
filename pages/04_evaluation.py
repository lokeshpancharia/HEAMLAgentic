"""Page 4: Evaluation — metrics, SHAP, parity plot, HTML report."""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import streamlit as st

from core.state import WorkflowState
from core.llm_client import create_llm_client
from core.theme import inject_theme, page_hero


def _get_state() -> WorkflowState:
    return st.session_state.workflow_state


def _log(msg: str):
    st.session_state.agent_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


inject_theme()
page_hero("Model Evaluation", "R² · MAE · RMSE with bootstrap CI · SHAP feature importance · Parity plot · HTML report", "📊")

state = _get_state()

# ── Prerequisites ─────────────────────────────────────────────────────────────
if not state.best_model_path:
    st.warning("No trained model available. Complete **AutoML** first.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Model", state.best_model_type or "—")
col2.metric("CV R²", f"{state.best_cv_r2:.3f}" if state.best_cv_r2 else "—")
col3.metric("CV RMSE", f"{state.best_cv_rmse:.2f}" if state.best_cv_rmse else "—")

# ── Run Evaluation ────────────────────────────────────────────────────────────
if st.button("📊 Run Evaluation Agent", type="primary", use_container_width=True):
    log_container = st.empty()
    logs: list[str] = []

    def stream_log(agent_name: str, msg: str):
        logs.append(f"[{agent_name}] {msg}")
        log_container.code("\n".join(logs[-20:]))

    with st.spinner("Running evaluation (SHAP analysis may take a moment)..."):
        try:
            llm = create_llm_client(state.llm_provider, state.llm_model)
            from agents.evaluation_agent import EvaluationAgent
            agent = EvaluationAgent(llm, state)
            agent.set_log_callback(stream_log)

            result = agent.evaluate(
                model_path=state.best_model_path,
                data_path=state.processed_data_path,
                target_column=state.target_property,
            )

            _log("Evaluation complete")
            st.success("Evaluation complete!")
            st.markdown("**Agent Summary:**")
            st.markdown(result)

            # Load results
            try:
                import json
                from config import REPORTS_DIR, PROCESSED_DIR
                import numpy as np
                import pandas as pd

                # Find latest report
                reports = sorted(REPORTS_DIR.glob(f"*_{state.target_property}_report.html"))
                if reports:
                    state.report_path = str(reports[-1])

                # Find predictions
                pred_files = sorted(PROCESSED_DIR.glob(f"predictions_{state.target_property}_*.parquet"))
                # Find SHAP values
                shap_files = sorted(PROCESSED_DIR.glob(f"shap_{state.target_property}_*.parquet"))

                # Display metrics
                if pred_files:
                    pred_df = pd.read_parquet(pred_files[-1])
                    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

                    y_true = pred_df["y_true"].values
                    y_pred = pred_df["y_pred"].values
                    r2 = r2_score(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

                    state.evaluation_metrics = {"r2": r2, "mae": mae, "rmse": rmse}

                    st.divider()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R²", f"{r2:.4f}")
                    c2.metric("MAE", f"{mae:.2f}")
                    c3.metric("RMSE", f"{rmse:.2f}")

                    # Parity plot
                    tab1, tab2, tab3 = st.tabs(["Parity Plot", "SHAP Features", "Residuals"])
                    with tab1:
                        try:
                            import plotly.express as px
                            import plotly.graph_objects as go

                            fig = px.scatter(
                                x=y_true, y=y_pred,
                                labels={"x": f"Actual {state.target_property}",
                                        "y": f"Predicted {state.target_property}"},
                                title="Predicted vs Actual",
                                opacity=0.7,
                            )
                            # Add perfect prediction line
                            lim = [min(y_true.min(), y_pred.min()),
                                   max(y_true.max(), y_pred.max())]
                            fig.add_trace(go.Scatter(x=lim, y=lim,
                                                     mode="lines",
                                                     line=dict(dash="dash", color="red"),
                                                     name="Perfect prediction"))
                            st.plotly_chart(fig, use_container_width=True)
                        except ImportError:
                            st.info("Install plotly for interactive charts")

                    with tab2:
                        if shap_files:
                            try:
                                shap_df = pd.read_parquet(shap_files[-1])
                                mean_shap = shap_df.abs().mean().sort_values(ascending=False)
                                top20 = mean_shap.head(20)
                                top_list = [{"feature": k, "mean_abs_shap": v}
                                            for k, v in top20.items()]
                                state.top_features = top_list

                                import plotly.express as px
                                fig2 = px.bar(
                                    x=top20.values,
                                    y=top20.index,
                                    orientation="h",
                                    title="Top 20 Features by SHAP Importance",
                                    labels={"x": "Mean |SHAP|", "y": "Feature"},
                                )
                                fig2.update_layout(yaxis={"autorange": "reversed"})
                                st.plotly_chart(fig2, use_container_width=True)
                            except Exception:
                                st.info("SHAP visualization unavailable")
                        else:
                            st.info("SHAP values not yet computed")

                    with tab3:
                        try:
                            residuals = y_true - y_pred
                            import plotly.express as px
                            fig3 = px.scatter(
                                x=y_pred, y=residuals,
                                labels={"x": "Predicted", "y": "Residual"},
                                title="Residual Plot",
                                opacity=0.6,
                            )
                            fig3.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig3, use_container_width=True)
                        except ImportError:
                            pass

            except Exception:
                pass  # Best-effort display

        except Exception as e:
            st.error(f"Agent error: {e}")
            _log(f"ERROR: {e}")

# ── Report Download ────────────────────────────────────────────────────────────
if state.report_path and Path(state.report_path).exists():
    st.divider()
    with open(state.report_path, "rb") as f:
        report_bytes = f.read()
    st.download_button(
        label="⬇️ Download HTML Report",
        data=report_bytes,
        file_name=Path(state.report_path).name,
        mime="text/html",
        use_container_width=True,
    )

# ── Agent log ─────────────────────────────────────────────────────────────────
if st.session_state.agent_logs:
    with st.expander("Agent Log", expanded=False):
        st.code("\n".join(st.session_state.agent_logs[-50:]))
