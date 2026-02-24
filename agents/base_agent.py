"""BaseAgent: agentic loop with provider-agnostic LLM client and tool dispatch."""

from __future__ import annotations
from typing import Any

from core.llm_client import BaseLLMClient
from core.tool_registry import ToolRegistry
from core.state import WorkflowState


class BaseAgent:
    """
    Provider-agnostic agentic loop.
    Subclasses define system_prompt, tool_definitions, and tool_callables.
    """

    name: str = "BaseAgent"
    system_prompt: str = "You are a helpful AI agent."

    def __init__(self, llm_client: BaseLLMClient, state: WorkflowState):
        self.llm_client = llm_client
        self.state = state
        self.tool_definitions: list[dict] = []
        self.registry = ToolRegistry({})
        self.messages: list[dict] = []
        self._log_callback = None  # optional streaming callback

    def set_tools(self, definitions: list[dict], callables: dict):
        self.tool_definitions = definitions
        self.registry = ToolRegistry(callables)

    def set_log_callback(self, callback):
        """Set a callback fn(agent_name, message) for real-time log streaming."""
        self._log_callback = callback

    def _log(self, message: str, level: str = "info"):
        self.state.log(self.name, message, level)
        if self._log_callback:
            self._log_callback(self.name, message)

    def _extract_text(self, content: Any) -> str:
        """Extract text from either Anthropic content blocks or normalized raw_content."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif hasattr(block, "type") and block.type == "text":
                    parts.append(block.text)
            return "\n".join(parts)
        return str(content)

    def run(self, task: str, max_iterations: int = 30) -> str:
        """
        Core agentic loop:
        - Feed task to LLM
        - Execute tool calls
        - Repeat until stop_reason == "end_turn"
        """
        self.messages = []  # fresh conversation per run
        self.messages.append({"role": "user", "content": task})
        self._log(f"Starting task: {task[:100]}...")

        for iteration in range(max_iterations):
            response = self.llm_client.chat(
                messages=self.messages,
                tools=self.tool_definitions,
                system=self.system_prompt,
            )

            # Append assistant turn to history
            self.messages.append({
                "role": "assistant",
                "content": response.raw_content,
            })

            if response.stop_reason == "end_turn":
                self._log("Task complete.")
                return response.text

            if response.stop_reason == "tool_use":
                tool_results = []
                for tc in response.tool_calls:
                    self._log(f"Calling tool: {tc.name}({list(tc.input.keys())})")
                    result = self.registry.dispatch(tc.name, tc.input)
                    self._log(f"Tool {tc.name} result: {str(result)[:200]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": str(result),
                    })
                self.messages.append({"role": "user", "content": tool_results})
            else:
                # Unexpected stop reason — treat as done
                break

        self._log("Max iterations reached.", level="warning")
        return response.text if response else "Agent reached max iterations without completing."
