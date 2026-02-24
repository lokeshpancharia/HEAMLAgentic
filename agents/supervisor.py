"""Orchestrator/Supervisor Agent: coordinates the full HEA ML workflow."""

from __future__ import annotations
import json

from agents.base_agent import BaseAgent
from agents.literature_agent import LiteratureMiningAgent
from agents.data_agent import DataProcessingAgent
from agents.automl_agent import AutoMLAgent
from agents.evaluation_agent import EvaluationAgent
from core.llm_client import BaseLLMClient, create_llm_client
from core.tool_registry import ToolRegistry
from core.state import WorkflowState


# ── Supervisor tool definitions ───────────────────────────────────────────────

SUPERVISOR_TOOL_DEFINITIONS = [
    {
        "name": "run_literature_agent",
        "description": (
            "Delegates to the Literature Mining Agent. Use this when you need to collect "
            "HEA composition-property data from Materials Project, AFLOW, arXiv papers, or PDFs. "
            "Returns a summary of data collected and the path to the merged raw data file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Specific instructions for the literature agent"},
                "elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Chemical elements to focus on",
                },
                "target_property": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "['materials_project', 'aflow', 'arxiv', 'pdf']",
                },
                "arxiv_query": {"type": "string"},
            },
            "required": ["task"],
        },
    },
    {
        "name": "run_data_agent",
        "description": (
            "Delegates to the Data Processing Agent. Use after data is available (either uploaded "
            "by user or collected by literature agent). Performs HEA feature engineering. "
            "Returns path to featurized .parquet file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "input_path": {"type": "string", "description": "Path to raw data file (CSV, JSON, or parquet)"},
                "target_column": {"type": "string"},
                "featurizer_set": {
                    "type": "string",
                    "description": "'minimal', 'standard', or 'comprehensive'",
                },
                "composition_column": {
                    "type": "string",
                    "description": "Name of the composition column in the data file",
                },
            },
            "required": ["task", "input_path", "target_column"],
        },
    },
    {
        "name": "run_automl_agent",
        "description": (
            "Delegates to the AutoML Agent. Searches for the best ML model and optimizes "
            "hyperparameters with Optuna. Returns model type, CV R² score, and path to saved model."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "data_path": {"type": "string", "description": "Path to featurized .parquet file"},
                "target_column": {"type": "string"},
                "n_trials": {"type": "integer", "description": "Optuna HPO trials per model (default 50)"},
                "cv_folds": {"type": "integer"},
                "task_type": {"type": "string", "description": "'regression' or 'classification'"},
            },
            "required": ["task", "data_path", "target_column"],
        },
    },
    {
        "name": "run_evaluation_agent",
        "description": (
            "Delegates to the Evaluation Agent. Generates SHAP explainability, metrics with CI, "
            "and an HTML report. Returns path to the HTML report and metric summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "model_path": {"type": "string"},
                "data_path": {"type": "string"},
                "target_column": {"type": "string"},
            },
            "required": ["task", "model_path", "data_path", "target_column"],
        },
    },
]


SUPERVISOR_SYSTEM_PROMPT = """You are the Orchestrator for HEAMLAgentic, a multi-agent system for High Entropy Alloy ML model discovery.

You coordinate four specialist agents to complete the full ML workflow.

WORKFLOW:
The workflow has two entry paths:
1. BYOD (Bring Your Own Data): raw_data_path is already provided → skip literature agent, start at run_data_agent
2. Literature collection: elements and target_property provided without raw_data_path → start at run_literature_agent

Standard execution order (skip step 1 if raw_data_path is already set):
1. run_literature_agent  → collects raw HEA data (ONLY if no raw_data_path)
2. run_data_agent        → engineers features from raw data
3. run_automl_agent      → trains and optimizes the best ML model
4. run_evaluation_agent  → evaluates model, generates SHAP + HTML report

RULES:
- Pass the output path from each agent as input to the next
- If any agent returns an ERROR, report it clearly with remediation suggestions
- After all steps complete, provide a concise summary:
  * Number of data points collected/used
  * Best model type and CV R² score
  * Top 3 most important features
  * Path to HTML evaluation report

Always be explicit about which step you're on and what each agent returned.
"""


class OrchestratorAgent(BaseAgent):
    """Supervisor that delegates to specialized sub-agents."""

    name = "OrchestratorAgent"
    system_prompt = SUPERVISOR_SYSTEM_PROMPT

    def __init__(self, llm_client: BaseLLMClient, state: WorkflowState,
                 log_callback=None):
        super().__init__(llm_client, state)
        self._log_callback = log_callback

        # Instantiate sub-agents (they share the same LLM client and state)
        self._lit_agent = LiteratureMiningAgent(llm_client, state)
        self._data_agent = DataProcessingAgent(llm_client, state)
        self._automl_agent = AutoMLAgent(llm_client, state)
        self._eval_agent = EvaluationAgent(llm_client, state)

        # Propagate log callback to sub-agents
        if log_callback:
            for agent in [self._lit_agent, self._data_agent,
                          self._automl_agent, self._eval_agent]:
                agent.set_log_callback(log_callback)

        # Wire tool definitions and callables
        callables = {
            "run_literature_agent": self._run_literature_agent,
            "run_data_agent": self._run_data_agent,
            "run_automl_agent": self._run_automl_agent,
            "run_evaluation_agent": self._run_evaluation_agent,
        }
        self.set_tools(SUPERVISOR_TOOL_DEFINITIONS, callables)

    # ── Meta-tool implementations ─────────────────────────────────────────────

    def _run_literature_agent(
        self,
        task: str,
        elements: list[str] | None = None,
        target_property: str | None = None,
        sources: list[str] | None = None,
        arxiv_query: str | None = None,
    ) -> str:
        self._log("Delegating to LiteratureMiningAgent...")
        result = self._lit_agent.collect(
            target_property=target_property or self.state.target_property or "properties",
            elements=elements or self.state.elements or [],
            sources=sources,
            arxiv_query=arxiv_query,
        )
        return result

    def _run_data_agent(
        self,
        task: str,
        input_path: str | None = None,
        target_column: str | None = None,
        featurizer_set: str = "standard",
        composition_column: str = "composition",
    ) -> str:
        self._log("Delegating to DataProcessingAgent...")
        # Use state raw_data_path if not provided
        path = input_path or self.state.raw_data_path
        if not path:
            return "ERROR: No input data path provided. Either upload data or run literature agent first."
        col = target_column or self.state.target_property
        if not col:
            return "ERROR: No target column specified."
        result = self._data_agent.process(
            input_path=path,
            target_column=col,
            featurizer_set=featurizer_set,
            composition_column=composition_column,
        )
        return result

    def _run_automl_agent(
        self,
        task: str,
        data_path: str | None = None,
        target_column: str | None = None,
        n_trials: int = 50,
        cv_folds: int = 5,
        task_type: str = "regression",
    ) -> str:
        self._log("Delegating to AutoMLAgent...")
        path = data_path or self.state.processed_data_path
        if not path:
            return "ERROR: No processed data path. Run data agent first."
        col = target_column or self.state.target_property
        result = self._automl_agent.find_best_model(
            data_path=path,
            target_column=col,
            n_trials=n_trials,
            cv_folds=cv_folds,
            task_type=task_type,
        )
        return result

    def _run_evaluation_agent(
        self,
        task: str,
        model_path: str | None = None,
        data_path: str | None = None,
        target_column: str | None = None,
    ) -> str:
        self._log("Delegating to EvaluationAgent...")
        m_path = model_path or self.state.best_model_path
        d_path = data_path or self.state.processed_data_path
        col = target_column or self.state.target_property
        if not m_path:
            return "ERROR: No model path. Run AutoML agent first."
        result = self._eval_agent.evaluate(
            model_path=m_path,
            data_path=d_path,
            target_column=col,
        )
        return result

    # ── High-level entry points ───────────────────────────────────────────────

    def run_byod_workflow(
        self,
        raw_data_path: str,
        target_column: str,
        composition_column: str = "composition",
        featurizer_set: str = "standard",
        n_trials: int = 50,
    ) -> str:
        """BYOD workflow: skip literature, start from data processing."""
        self.state.raw_data_path = raw_data_path
        self.state.target_property = target_column
        self.state.data_entry_mode = "byod"

        task = f"""Run the complete HEA ML workflow using the provided dataset.

Dataset: {raw_data_path}
Target property: {target_column}
Composition column: {composition_column}
Featurizer set: {featurizer_set}
HPO trials: {n_trials}

The raw data is already provided — DO NOT run the literature agent.
Start directly with run_data_agent, then run_automl_agent, then run_evaluation_agent.
Report the complete results at the end."""

        return self.run(task)

    def run_literature_workflow(
        self,
        target_property: str,
        elements: list[str],
        sources: list[str] | None = None,
        arxiv_query: str | None = None,
        featurizer_set: str = "standard",
        n_trials: int = 50,
    ) -> str:
        """Full literature-to-model workflow."""
        self.state.target_property = target_property
        self.state.elements = elements
        self.state.data_entry_mode = "literature"

        task = f"""Run the complete HEA ML discovery workflow from literature collection to model evaluation.

Target property: {target_property}
Elements: {elements}
Data sources: {sources or ['materials_project', 'aflow', 'arxiv']}
arXiv query: {arxiv_query or 'high entropy alloy ' + target_property}
Featurizer set: {featurizer_set}
HPO trials: {n_trials}

Follow the full workflow:
1. run_literature_agent to collect data
2. run_data_agent to process and featurize
3. run_automl_agent to find the best model
4. run_evaluation_agent to evaluate and generate report

Provide a comprehensive summary at the end."""

        return self.run(task)
