"""Page 5: Predict — single composition or batch predictions."""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import streamlit as st

from core.state import WorkflowState


def _get_state() -> WorkflowState:
    return st.session_state.workflow_state


st.title("🔮 Predict")
st.caption("Predict material properties for new HEA compositions")

state = _get_state()

# ── Prerequisites ─────────────────────────────────────────────────────────────
if not state.best_model_path:
    st.warning("No trained model available. Complete the **AutoML** step first.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Model", state.best_model_type or "—")
col2.metric("Target", state.target_property or "—")
col3.metric("CV R²", f"{state.best_cv_r2:.3f}" if state.best_cv_r2 else "—")

st.divider()

# ── Prediction functions ───────────────────────────────────────────────────────
def predict_composition(composition: str) -> dict | None:
    """Predict property for a single composition."""
    try:
        import joblib
        import json
        import pandas as pd
        import numpy as np

        model = joblib.load(state.best_model_path)
        meta_path = Path(state.best_model_path).with_suffix(".json")

        if not meta_path.exists():
            st.error("Model metadata not found. Cannot determine feature pipeline.")
            return None

        with open(meta_path) as f:
            meta = json.load(f)

        feature_names = meta["feature_names"]
        target_col = meta["target_column"]
        featurizer_set = state.featurizer_set or "standard"

        # Featurize the single composition
        from tools.cleaning_tools import clean_dataset
        from tools.feature_tools import featurize_compositions

        # Create temp dataframe
        temp_df = pd.DataFrame({"composition": [composition], target_col: [0.0]})

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        from config import RAW_DIR
        temp_path = RAW_DIR / f"temp_predict_{ts}.parquet"
        temp_df.to_parquet(temp_path, index=False)

        # Featurize
        result_json = featurize_compositions(
            input_path=str(temp_path),
            target_column=target_col,
            featurizer_set=featurizer_set,
        )

        import json as _json
        result = _json.loads(result_json)

        if "featurized_path" not in result:
            # Clean up
            temp_path.unlink(missing_ok=True)
            return {"error": result.get("error", "Featurization failed")}

        feat_df = pd.read_parquet(result["featurized_path"])

        # Align features
        for col in feature_names:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        X = feat_df[feature_names].values.astype(float)

        # Predict
        y_pred = model.predict(X)[0]

        # SHAP for this composition
        shap_values = None
        try:
            import shap
            from sklearn.pipeline import Pipeline
            actual_model = model.steps[-1][1] if isinstance(model, Pipeline) else model
            explainer = shap.TreeExplainer(actual_model)
            sv = explainer.shap_values(X)
            shap_pairs = sorted(
                zip(feature_names, sv[0].tolist()),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:10]
            shap_values = shap_pairs
        except Exception:
            pass

        # Clean up temp files
        temp_path.unlink(missing_ok=True)
        Path(result["featurized_path"]).unlink(missing_ok=True)

        return {
            "composition": composition,
            "predicted_value": float(y_pred),
            "target_property": target_col,
            "shap_values": shap_values,
        }

    except Exception as e:
        return {"error": str(e)}


# ── Single Prediction ─────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["Single Composition", "Batch Predictions"])

with tab_single:
    st.markdown("Enter an HEA composition to predict its property.")
    st.markdown("**Format examples:** `Al0.5CoCrFeNi`, `CrMnFeCoNi`, `Al0.3Co1.0Cr1.0Fe1.0Ni1.0`")

    composition_input = st.text_input(
        "Composition",
        placeholder="e.g., Al0.5CoCrFeNi",
        help="Enter composition in Cantor-notation or molar fraction notation",
    )

    if st.button("🔮 Predict", type="primary", use_container_width=False,
                 disabled=not composition_input):
        with st.spinner(f"Featurizing and predicting {composition_input}..."):
            result = predict_composition(composition_input.strip())

        if result and "error" not in result:
            st.success(f"**Predicted {result['target_property']}: {result['predicted_value']:.2f}**")

            # SHAP waterfall
            if result.get("shap_values"):
                st.subheader("Feature Contributions (SHAP)")
                try:
                    import plotly.graph_objects as go

                    features = [s[0] for s in result["shap_values"]]
                    values = [s[1] for s in result["shap_values"]]
                    colors = ["#1a73e8" if v > 0 else "#ea4335" for v in values]

                    fig = go.Figure(go.Bar(
                        x=values,
                        y=features,
                        orientation="h",
                        marker_color=colors,
                    ))
                    fig.update_layout(
                        title="Top 10 Feature Contributions",
                        xaxis_title="SHAP Value",
                        yaxis={"autorange": "reversed"},
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.write("**Top contributing features:**")
                    for feat, val in result["shap_values"]:
                        direction = "↑" if val > 0 else "↓"
                        st.write(f"- {feat}: {val:+.4f} {direction}")
        elif result and "error" in result:
            st.error(f"Prediction failed: {result['error']}")
            st.info("Ensure the composition format is correct, e.g., `Al0.5CoCrFeNi`")

# ── Batch Predictions ─────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("Upload a CSV with a **`composition`** column to predict properties for multiple compositions.")

    batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

    if batch_file:
        import pandas as pd
        batch_df = pd.read_csv(batch_file)

        if "composition" not in batch_df.columns:
            comp_col = st.selectbox("Select composition column", batch_df.columns)
            batch_df = batch_df.rename(columns={comp_col: "composition"})

        st.dataframe(batch_df.head(), use_container_width=True)
        st.info(f"{len(batch_df)} compositions to predict")

        if st.button("🔮 Run Batch Predictions", type="primary"):
            progress_bar = st.progress(0)
            results_list = []

            for i, row in batch_df.iterrows():
                result = predict_composition(str(row["composition"]))
                if result and "error" not in result:
                    results_list.append({
                        "composition": row["composition"],
                        f"predicted_{state.target_property}": result["predicted_value"],
                    })
                else:
                    results_list.append({
                        "composition": row["composition"],
                        f"predicted_{state.target_property}": None,
                    })
                progress_bar.progress((i + 1) / len(batch_df))

            results_df = pd.DataFrame(results_list)
            st.dataframe(results_df, use_container_width=True)

            # Download
            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_bytes,
                file_name=f"predictions_{state.target_property}_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
            )
