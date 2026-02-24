"""WorkflowState — single source of truth shared across all agents and Streamlit pages."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import json
import os


@dataclass
class WorkflowState:
    # ── LLM Provider ─────────────────────────────────────────────────────────
    llm_provider: str = "anthropic"   # "anthropic" | "openai" | "gemini"
    llm_model: str = "claude-sonnet-4-6"

    # ── Phase 1: Data Entry ───────────────────────────────────────────────────
    target_property: Optional[str] = None
    elements: list[str] = field(default_factory=list)
    data_entry_mode: str = "byod"   # "byod" | "literature"

    # BYOD path
    raw_data_path: Optional[str] = None          # uploaded CSV/JSON saved here
    n_raw_samples: Optional[int] = None

    # Literature path
    collection_sources: list[str] = field(default_factory=list)
    arxiv_query: Optional[str] = None

    # ── Phase 2: Processing ───────────────────────────────────────────────────
    processed_data_path: Optional[str] = None
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    feature_names: list[str] = field(default_factory=list)
    featurizer_set: str = "standard"

    # ── Phase 3: AutoML ───────────────────────────────────────────────────────
    model_leaderboard: list[dict[str, Any]] = field(default_factory=list)
    best_model_type: Optional[str] = None
    best_model_path: Optional[str] = None
    best_params: Optional[dict[str, Any]] = None
    best_cv_r2: Optional[float] = None
    best_cv_rmse: Optional[float] = None
    model_metadata_path: Optional[str] = None

    # ── Phase 4: Evaluation ───────────────────────────────────────────────────
    evaluation_metrics: Optional[dict[str, Any]] = None
    shap_values_path: Optional[str] = None
    report_path: Optional[str] = None
    top_features: list[dict[str, Any]] = field(default_factory=list)

    # ── Run logs ──────────────────────────────────────────────────────────────
    run_logs: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # ── Helpers ───────────────────────────────────────────────────────────────
    def log(self, agent: str, message: str, level: str = "info"):
        self.run_logs.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "level": level,
            "message": message,
        })

    def reset(self):
        """Reset to initial state while preserving LLM provider selection."""
        provider, model = self.llm_provider, self.llm_model
        self.__init__()
        self.llm_provider = provider
        self.llm_model = model

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items()
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "WorkflowState":
        with open(path) as f:
            data = json.load(f)
        state = cls()
        for k, v in data.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state
