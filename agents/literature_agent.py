"""Literature Mining Agent: collects HEA data from MP, AFLOW, arXiv, and PDFs."""

from __future__ import annotations
import json

from agents.base_agent import BaseAgent
from core.llm_client import BaseLLMClient
from core.state import WorkflowState
from tools.mp_tools import MP_TOOL_DEFINITIONS, MP_TOOL_CALLABLES
from tools.aflow_tools import AFLOW_TOOL_DEFINITIONS, AFLOW_TOOL_CALLABLES
from tools.arxiv_tools import ARXIV_TOOL_DEFINITIONS, ARXIV_TOOL_CALLABLES
from tools.pdf_tools import PDF_TOOL_DEFINITIONS, PDF_TOOL_CALLABLES
from tools.cleaning_tools import CLEANING_TOOL_DEFINITIONS, CLEANING_TOOL_CALLABLES


LITERATURE_SYSTEM_PROMPT = """You are the Literature Mining Agent for HEAMLAgentic, a system for High Entropy Alloy (HEA) ML discovery.

Your job is to collect HEA composition-property data from multiple sources and save it to disk.

AVAILABLE TOOLS:
1. query_materials_project — Query Materials Project database for alloy properties
2. query_aflow — Query AFLOW database
3. search_arxiv_papers — Search arXiv for relevant papers
4. extract_data_from_pdf — Download and extract data from a paper PDF
5. merge_raw_data — Merge all collected JSON files into a single dataset

WORKFLOW:
1. Query structured databases (Materials Project, AFLOW) with the provided elements
2. Search arXiv for papers about the target property
3. For the top 5-10 most relevant papers, call extract_data_from_pdf
4. After all collection is done, call merge_raw_data to unify all records
5. Report total records collected, sources used, and the merged file path

RULES:
- Always query Materials Project first (highest quality data)
- For arXiv: search using terms like "high entropy alloy [property] machine learning"
- Only extract PDFs that appear highly relevant (check abstract first)
- Call merge_raw_data at the end with the target_column name
- Report final data count and the merged CSV path in your final response
"""

ALL_TOOL_DEFINITIONS = (
    MP_TOOL_DEFINITIONS
    + AFLOW_TOOL_DEFINITIONS
    + ARXIV_TOOL_DEFINITIONS
    + PDF_TOOL_DEFINITIONS
    + CLEANING_TOOL_DEFINITIONS
)

ALL_TOOL_CALLABLES = {
    **MP_TOOL_CALLABLES,
    **AFLOW_TOOL_CALLABLES,
    **ARXIV_TOOL_CALLABLES,
    **PDF_TOOL_CALLABLES,
    **CLEANING_TOOL_CALLABLES,
}


class LiteratureMiningAgent(BaseAgent):
    name = "LiteratureMiningAgent"
    system_prompt = LITERATURE_SYSTEM_PROMPT

    def __init__(self, llm_client: BaseLLMClient, state: WorkflowState):
        super().__init__(llm_client, state)
        self.set_tools(ALL_TOOL_DEFINITIONS, ALL_TOOL_CALLABLES)

    def collect(
        self,
        target_property: str,
        elements: list[str],
        sources: list[str] | None = None,
        arxiv_query: str | None = None,
    ) -> str:
        """Run the literature mining task."""
        sources = sources or ["materials_project", "aflow", "arxiv"]
        if arxiv_query is None:
            elements_str = "".join(elements)
            arxiv_query = f"high entropy alloy {elements_str} {target_property} machine learning"

        task = f"""Collect HEA composition-property data for the following:
- Target property: {target_property}
- Elements of interest: {elements}
- Data sources to use: {sources}
- arXiv search query: "{arxiv_query}"

Please collect data from all specified sources, then merge all collected data into a single file.
Report the total number of records collected and the path to the merged dataset."""

        result = self.run(task)

        # Update state
        self.state.collection_sources = sources
        self.state.elements = elements
        self.state.target_property = target_property

        return result
