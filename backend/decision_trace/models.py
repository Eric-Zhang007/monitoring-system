from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DecisionTrace(BaseModel):
    ts: datetime = Field(default_factory=_utcnow)
    decision_id: str
    symbol: str
    action: str
    target_pos: float
    delta_pos: float
    exec_style: str = "marketable_limit"
    deadline_s: int = 30
    slices: int = 1
    mu: Dict[str, float] = Field(default_factory=dict)
    sigma: Dict[str, float] = Field(default_factory=dict)
    quantiles: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    direction_prob: Dict[str, float] = Field(default_factory=dict)
    expert_weights: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    regime_probs: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    cost: Dict[str, float] = Field(default_factory=dict)
    cost_breakdown: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    score: Dict[str, float] = Field(default_factory=dict)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    risk: Dict[str, Any] = Field(default_factory=dict)
    account: Dict[str, Any] = Field(default_factory=dict)
    position: Dict[str, Any] = Field(default_factory=dict)
    analyst: Dict[str, Any] = Field(default_factory=dict)
    reason_codes: List[str] = Field(default_factory=list)
