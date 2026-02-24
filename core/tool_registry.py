"""Maps tool names to Python callables for agent dispatch."""

from __future__ import annotations
from typing import Any, Callable


class ToolRegistry:
    def __init__(self, tool_callables: dict[str, Callable]):
        """
        Args:
            tool_callables: dict mapping tool name → Python callable
        """
        self._registry: dict[str, Callable] = tool_callables

    def dispatch(self, name: str, inputs: dict[str, Any]) -> Any:
        """Execute a tool by name with given inputs."""
        if name not in self._registry:
            return f"ERROR: Unknown tool '{name}'. Available: {list(self._registry.keys())}"
        try:
            return self._registry[name](**inputs)
        except Exception as e:
            return f"ERROR in tool '{name}': {type(e).__name__}: {e}"

    def register(self, name: str, callable_fn: Callable):
        self._registry[name] = callable_fn
