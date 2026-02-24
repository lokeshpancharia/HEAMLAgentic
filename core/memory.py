"""Simple JSON file-backed memory for agent context persistence."""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Any


class AgentMemory:
    """Lightweight key-value store persisted as JSON. One file per agent."""

    def __init__(self, agent_name: str, base_dir: str = "data"):
        self.agent_name = agent_name
        self.path = os.path.join(base_dir, f".{agent_name}_memory.json")
        self._store: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._store, f, indent=2, default=str)

    def set(self, key: str, value: Any):
        self._store[key] = value
        self._store[f"_updated_{key}"] = datetime.now().isoformat()
        self.save()

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def clear(self):
        self._store = {}
        if os.path.exists(self.path):
            os.remove(self.path)
