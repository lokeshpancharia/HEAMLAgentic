"""Evaluation Agent: metrics, SHAP analysis, and HTML report generation."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from core.llm_client import BaseLLMClient
from core.state import WorkflowState
from tools.evaluation_tools import EVALUATION_TOOL_DEFINITIONS, EVALUATION_TOOL_CALLABLES


EVALUATION_SYSTEM_PROMPT = """You are the Evaluation Agent for HEAMLAgentic. Your job is to thoroughly evaluate a trained ML model for HEA property prediction.

AVAILABLE TOOLS:
1. compute_metrics — Compute R², MAE, RMSE, MAPE with 95% bootstrap CI
2. generate_shap_analysis — Compute SHAP feature importances
3. generate_html_report — Generate a self-contained HTML evaluation report

WORKFLOW (always follow this order):
1. Call compute_metrics to get all performance metrics
2. Call generate_shap_analysis to identify important features
3. Call generate_html_report with the metrics and top_features from steps 1 & 2
4. Summarize results clearly

YOUR FINAL RESPONSE must include:
- R² score (and 95% CI)
- MAE and RMSE (with units if known)
- Top 5 most important features by SHAP value
- Path to the HTML report
- Qualitative interpretation: is the model good enough for materials screening?

For context: R² > 0.85 is excellent for HEA ML models. R² > 0.7 is acceptable for screening.
Top SHAP features reveal which elemental properties drive the target property most.
"""


class EvaluationAgent(BaseAgent):
    name = "EvaluationAgent"
    system_prompt = EVALUATION_SYSTEM_PROMPT

    def __init__(self, llm_client: BaseLLMClient, state: WorkflowState):
        super().__init__(llm_client, state)
        self.set_tools(EVALUATION_TOOL_DEFINITIONS, EVALUATION_TOOL_CALLABLES)

    def evaluate(
        self,
        model_path: str,
        data_path: str,
        target_column: str,
    ) -> str:
        """Run complete evaluation pipeline."""
        task = f"""Evaluate the trained model for {target_column} prediction:
- Model path: {model_path}
- Data path: {data_path}
- Target column: {target_column}

1. Compute all metrics with bootstrap CI
2. Generate SHAP analysis for feature importance
3. Generate the HTML evaluation report
4. Provide a comprehensive summary with qualitative interpretation"""

        return self.run(task)
