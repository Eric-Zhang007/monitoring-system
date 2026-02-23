from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RiskRegime(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


def _default_soft_penalty_factors() -> Dict[str, Any]:
    return {
        "pos_scale": 1.0,
        "band_scale": 1.0,
        "cost_scale": 1.0,
        "exec_style_bias": "neutral",
    }


class RiskState(BaseModel):
    regime: RiskRegime = RiskRegime.GREEN
    hard_limits_ok: bool = True
    soft_penalty_factors: Dict[str, Any] = Field(default_factory=_default_soft_penalty_factors)
    reason_codes: List[str] = Field(default_factory=list)
