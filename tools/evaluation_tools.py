"""Model evaluation tools: metrics, SHAP, plots, HTML reports."""

from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from config import REPORTS_DIR, PROCESSED_DIR


EVALUATION_TOOL_DEFINITIONS = [
    {
        "name": "compute_metrics",
        "description": (
            "Compute regression metrics (R², MAE, RMSE, MAPE) with 95% bootstrap confidence intervals "
            "using cross-validation on the best trained model."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Path to .joblib model"},
                "data_path": {"type": "string", "description": "Path to featurized .parquet"},
                "target_column": {"type": "string"},
                "cv_folds": {"type": "integer"},
            },
            "required": ["model_path", "data_path", "target_column"],
        },
    },
    {
        "name": "generate_shap_analysis",
        "description": (
            "Compute SHAP values for the model and return top feature importances. "
            "Saves SHAP values to disk for visualization in Streamlit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "data_path": {"type": "string"},
                "target_column": {"type": "string"},
                "max_samples": {
                    "type": "integer",
                    "description": "Max samples for SHAP computation (default 500 for speed)",
                },
            },
            "required": ["model_path", "data_path", "target_column"],
        },
    },
    {
        "name": "generate_html_report",
        "description": (
            "Generate a self-contained HTML evaluation report with all metrics, "
            "feature importances, and summary. Returns the report file path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "data_path": {"type": "string"},
                "target_column": {"type": "string"},
                "metrics": {"type": "object", "description": "Metrics dict from compute_metrics"},
                "top_features": {
                    "type": "array",
                    "description": "Top features list from generate_shap_analysis",
                },
            },
            "required": ["model_path", "data_path", "target_column"],
        },
    },
]


def _load_model_and_data(model_path: str, data_path: str, target_column: str):
    import joblib
    import pandas as pd
    import numpy as np

    model = joblib.load(model_path)
    df = pd.read_parquet(data_path)
    feature_cols = [c for c in df.columns if c not in ("composition", target_column)]
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_column].values.astype(np.float64)
    return model, X, y, feature_cols, df


def compute_metrics(
    model_path: str,
    data_path: str,
    target_column: str,
    cv_folds: int = 5,
) -> str:
    """Compute CV metrics with bootstrap confidence intervals."""
    try:
        import numpy as np
        from sklearn.model_selection import cross_val_predict, KFold
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    except ImportError:
        return "ERROR: scikit-learn not installed"

    try:
        model, X, y, _, _ = _load_model_and_data(model_path, data_path, target_column)
    except Exception as e:
        return f"ERROR loading model/data: {e}"

    # Cross-validated predictions
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=kf)

    r2 = float(r2_score(y, y_pred))
    mae = float(mean_absolute_error(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    # MAPE (avoid division by zero)
    nonzero = y != 0
    mape = float(np.mean(np.abs((y[nonzero] - y_pred[nonzero]) / y[nonzero])) * 100) if nonzero.any() else None

    # Bootstrap CI on R²
    n_boot = 500
    boot_r2 = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        boot_r2.append(r2_score(y[idx], y_pred[idx]))
    r2_ci_low = float(np.percentile(boot_r2, 2.5))
    r2_ci_high = float(np.percentile(boot_r2, 97.5))

    # Save predictions for parity plot
    import pandas as pd
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = PROCESSED_DIR / f"predictions_{target_column}_{ts}.parquet"
    pd.DataFrame({"y_true": y, "y_pred": y_pred}).to_parquet(pred_path, index=False)

    return json.dumps({
        "r2": r2,
        "r2_ci_95": [r2_ci_low, r2_ci_high],
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "n_samples": len(y),
        "cv_folds": cv_folds,
        "predictions_path": str(pred_path),
        "target_stats": {
            "mean": float(y.mean()),
            "std": float(y.std()),
        },
    }, default=str)


def generate_shap_analysis(
    model_path: str,
    data_path: str,
    target_column: str,
    max_samples: int = 500,
) -> str:
    """Compute SHAP values and return top feature importances."""
    try:
        import shap
        import numpy as np
        import pandas as pd
    except ImportError:
        return "ERROR: shap not installed. Run: pip install shap"

    try:
        model, X, y, feature_cols, df = _load_model_and_data(model_path, data_path, target_column)
    except Exception as e:
        return f"ERROR: {e}"

    # Subsample for speed
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # Choose explainer based on model type
    try:
        # Try TreeExplainer first (fast for tree models)
        from sklearn.pipeline import Pipeline
        actual_model = model.steps[-1][1] if isinstance(model, Pipeline) else model
        explainer = shap.TreeExplainer(actual_model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        try:
            # Fallback to KernelExplainer (slower, works for any model)
            background = shap.sample(X_sample, min(100, len(X_sample)))
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_sample[:50])  # limit for speed
        except Exception as e:
            return f"ERROR computing SHAP values: {e}"

    # Mean absolute SHAP per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        zip(feature_cols, mean_shap.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    # Save SHAP values
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shap_path = PROCESSED_DIR / f"shap_{target_column}_{ts}.parquet"
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.to_parquet(shap_path, index=False)

    top_features = [
        {"feature": name, "mean_abs_shap": val}
        for name, val in feature_importance[:20]
    ]

    return json.dumps({
        "top_features": top_features,
        "shap_path": str(shap_path),
        "n_samples_used": len(X_sample),
        "n_features": len(feature_cols),
    }, default=str)


def generate_html_report(
    model_path: str,
    data_path: str,
    target_column: str,
    metrics: dict | None = None,
    top_features: list | None = None,
) -> str:
    """Generate self-contained HTML evaluation report."""
    import json as _json
    from jinja2 import Template

    # Load metadata if available
    meta_path = Path(model_path).with_suffix(".json")
    metadata: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = _json.load(f)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"{ts}_{target_column}_report.html"

    html_template = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>HEAMLAgentic — {{ target_column }} Model Report</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }
  h1 { color: #1a73e8; }
  h2 { color: #444; border-bottom: 2px solid #eee; padding-bottom: 8px; }
  .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }
  .metric-card { background: #f8f9fa; border-radius: 8px; padding: 16px; text-align: center; }
  .metric-value { font-size: 2em; font-weight: bold; color: #1a73e8; }
  .metric-label { color: #666; font-size: 0.85em; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; margin: 16px 0; }
  th { background: #1a73e8; color: white; padding: 10px; text-align: left; }
  td { padding: 8px 10px; border-bottom: 1px solid #eee; }
  tr:hover { background: #f5f5f5; }
  .bar { background: #1a73e8; height: 16px; border-radius: 4px; }
  .section { margin: 32px 0; }
  .badge { display: inline-block; background: #e8f0fe; color: #1a73e8; border-radius: 12px; padding: 4px 12px; font-size: 0.85em; }
</style>
</head>
<body>
<h1>HEAMLAgentic Evaluation Report</h1>
<p>Property: <strong>{{ target_column }}</strong> &nbsp;&nbsp;
   Model: <span class="badge">{{ model_type }}</span> &nbsp;&nbsp;
   Generated: {{ timestamp }}</p>

<div class="section">
<h2>Model Performance</h2>
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-value">{{ "%.3f"|format(r2) }}</div>
    <div class="metric-label">R² Score</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{{ "%.2f"|format(mae) }}</div>
    <div class="metric-label">MAE</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{{ "%.2f"|format(rmse) }}</div>
    <div class="metric-label">RMSE</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{{ n_samples }}</div>
    <div class="metric-label">Training Samples</div>
  </div>
</div>
</div>

{% if top_features %}
<div class="section">
<h2>Top 20 Most Important Features (SHAP)</h2>
<table>
<tr><th>#</th><th>Feature</th><th>Mean |SHAP|</th><th>Importance</th></tr>
{% set max_shap = top_features[0].mean_abs_shap %}
{% for f in top_features %}
<tr>
  <td>{{ loop.index }}</td>
  <td>{{ f.feature }}</td>
  <td>{{ "%.4f"|format(f.mean_abs_shap) }}</td>
  <td><div class="bar" style="width:{{ (f.mean_abs_shap / max_shap * 200)|int }}px"></div></td>
</tr>
{% endfor %}
</table>
</div>
{% endif %}

{% if metadata %}
<div class="section">
<h2>Model Configuration</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
{% for k, v in metadata.items() %}
{% if k not in ('feature_names', 'data_path', 'model_path', 'trained_at') %}
<tr><td>{{ k }}</td><td>{{ v }}</td></tr>
{% endif %}
{% endfor %}
</table>
</div>
{% endif %}

<p style="color:#999;font-size:0.8em;text-align:center;margin-top:40px">
  Generated by HEAMLAgentic &mdash; Multi-Agent HEA ML Discovery System
</p>
</body>
</html>"""

    template = Template(html_template)
    html = template.render(
        target_column=target_column,
        model_type=metadata.get("model_type", "Unknown"),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        r2=metrics.get("r2", 0) if metrics else 0,
        mae=metrics.get("mae", 0) if metrics else 0,
        rmse=metrics.get("rmse", 0) if metrics else 0,
        n_samples=metrics.get("n_samples", metadata.get("n_samples", 0)) if metrics else metadata.get("n_samples", 0),
        top_features=top_features or [],
        metadata=metadata,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return json.dumps({
        "report_path": str(report_path),
        "model_type": metadata.get("model_type", "Unknown"),
        "target_column": target_column,
    }, default=str)


EVALUATION_TOOL_CALLABLES = {
    "compute_metrics": compute_metrics,
    "generate_shap_analysis": generate_shap_analysis,
    "generate_html_report": generate_html_report,
}
