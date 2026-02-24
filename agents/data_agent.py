"""Data Processing Agent: feature engineering and cleaning for HEA datasets."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from core.llm_client import BaseLLMClient
from core.state import WorkflowState
from tools.cleaning_tools import CLEANING_TOOL_DEFINITIONS, CLEANING_TOOL_CALLABLES
from tools.feature_tools import FEATURE_TOOL_DEFINITIONS, FEATURE_TOOL_CALLABLES


DATA_SYSTEM_PROMPT = """You are the Data Processing Agent for HEAMLAgentic, specializing in HEA feature engineering.

Your job is to take a raw dataset of HEA compositions and properties, clean it, and featurize the compositions into ML-ready numerical features.

AVAILABLE TOOLS:
1. clean_dataset — Remove duplicates, outliers, impute missing values
2. featurize_compositions — Convert composition strings to numerical features using matminer

WORKFLOW:
1. First call clean_dataset on the input file to clean and validate the data
2. Then call featurize_compositions on the cleaned file to generate ML features
3. Report the final dataset: n_samples, n_features, feature names preview, and the featurized file path

COMPOSITION FORMAT:
- Accept: 'Al0.5CoCrFeNi', 'CrMnFeCoNi', 'Al0.3Co1.0Cr1.0Fe1.0Ni1.0'
- The featurizer handles these automatically via pymatgen

FEATURIZER SETS:
- 'minimal': ~20 features (Meredig), fastest
- 'standard': ~80 features (ElementProperty + YangSolidSolution + ValenceOrbital), recommended for HEA
- 'comprehensive': ~200 features, most complete but slower

Default to 'standard' unless the user specifies otherwise.

Report the final featurized file path and dataset statistics in your response.
"""

ALL_TOOL_DEFINITIONS = CLEANING_TOOL_DEFINITIONS + FEATURE_TOOL_DEFINITIONS
ALL_TOOL_CALLABLES = {**CLEANING_TOOL_CALLABLES, **FEATURE_TOOL_CALLABLES}


class DataProcessingAgent(BaseAgent):
    name = "DataProcessingAgent"
    system_prompt = DATA_SYSTEM_PROMPT

    def __init__(self, llm_client: BaseLLMClient, state: WorkflowState):
        super().__init__(llm_client, state)
        self.set_tools(ALL_TOOL_DEFINITIONS, ALL_TOOL_CALLABLES)

    def process(
        self,
        input_path: str,
        target_column: str,
        featurizer_set: str = "standard",
        composition_column: str = "composition",
    ) -> str:
        """Run the data processing pipeline."""
        task = f"""Process the HEA dataset for ML training:
- Input file: {input_path}
- Target property column: {target_column}
- Composition column: {composition_column}
- Featurizer set: {featurizer_set}

Steps:
1. Clean the dataset (remove duplicates, outliers, impute missing values)
2. Featurize the cleaned dataset using matminer with the '{featurizer_set}' featurizer set
3. Report: n_samples, n_features, and the path to the featurized .parquet file"""

        result = self.run(task)
        self.state.featurizer_set = featurizer_set
        return result
