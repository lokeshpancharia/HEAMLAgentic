"""AutoML Agent: model search and Optuna hyperparameter optimization."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from core.llm_client import BaseLLMClient
from core.state import WorkflowState
from tools.ml_tools import ML_TOOL_DEFINITIONS, ML_TOOL_CALLABLES
from config import AUTOML_MODELS


AUTOML_SYSTEM_PROMPT = f"""You are the AutoML Agent for HEAMLAgentic, specializing in finding the best ML model for HEA property prediction.

Your job is to search through multiple model types, optimize hyperparameters, and train the final best model.

AVAILABLE TOOLS:
1. run_model_comparison — Quick CV comparison of all models with default params
2. optimize_hyperparameters — Optuna HPO on a specific model
3. train_final_model — Train the winner on full data and save to disk

WORKFLOW:
1. Call run_model_comparison to get the leaderboard of all models
2. Call optimize_hyperparameters on the TOP 2 models from the leaderboard
3. Compare optimized results and select the best overall model
4. Call train_final_model with the best model type and its best params
5. Report: best model type, CV R², CV RMSE, model file path

AVAILABLE MODELS: {AUTOML_MODELS}

RULES:
- Always run model comparison first before HPO
- Optimize at least 2 models to ensure you find the true best
- Use the best_params from optimize_hyperparameters when calling train_final_model
- In your final response, clearly state:
  * Best model type
  * CV R² score
  * CV RMSE
  * Path to the saved model file

For regression tasks, optimize for minimum RMSE. R² > 0.85 is excellent for materials science.
"""


class AutoMLAgent(BaseAgent):
    name = "AutoMLAgent"
    system_prompt = AUTOML_SYSTEM_PROMPT

    def __init__(self, llm_client: BaseLLMClient, state: WorkflowState):
        super().__init__(llm_client, state)
        self.set_tools(ML_TOOL_DEFINITIONS, ML_TOOL_CALLABLES)

    def find_best_model(
        self,
        data_path: str,
        target_column: str,
        n_trials: int = 50,
        cv_folds: int = 5,
        task_type: str = "regression",
        models: list[str] | None = None,
    ) -> str:
        """Run the full AutoML pipeline."""
        models_str = str(models or AUTOML_MODELS)
        task = f"""Find the best ML model for predicting {target_column}:
- Data file: {data_path}
- Target column: {target_column}
- Task type: {task_type}
- Models to evaluate: {models_str}
- CV folds: {cv_folds}
- HPO trials per model: {n_trials}

Run the full AutoML pipeline:
1. Compare all models quickly
2. Optimize the top 2 models
3. Train the best final model

Report the best model type, CV R², CV RMSE, and saved model path."""

        return self.run(task)
