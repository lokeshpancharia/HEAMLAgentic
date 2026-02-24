"""Pydantic models for inter-agent data exchange."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    name: str
    id: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "tool_use" | "stop"
    raw_content: Any  # provider-specific raw content for message history


@dataclass
class CollectionResult:
    n_samples: int
    raw_data_path: str
    sources_used: list[str]
    summary: str


@dataclass
class ProcessingResult:
    n_samples: int
    n_features: int
    processed_data_path: str
    feature_names: list[str]
    target_column: str
    summary: str


@dataclass
class AutoMLResult:
    best_model_type: str
    best_model_path: str
    best_params: dict[str, Any]
    best_cv_r2: float
    best_cv_rmse: float
    leaderboard: list[dict[str, Any]]
    summary: str


@dataclass
class EvaluationResult:
    r2: float
    mae: float
    rmse: float
    mape: Optional[float]
    top_features: list[dict[str, Any]]
    report_path: str
    shap_values_path: Optional[str]
    summary: str
