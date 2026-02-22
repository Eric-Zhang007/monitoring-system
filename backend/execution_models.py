from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


DecisionStatus = Literal["created", "running", "completed", "failed"]
ParentOrderStatus = Literal["submitted", "executing", "filled", "partially_filled", "rejected", "canceled"]
ChildOrderStatus = Literal["new", "submitted", "filled", "partially_filled", "canceled", "rejected", "expired"]


class ExecutionLifecycleEvent(BaseModel):
    event: str
    status: str
    time: str
    metrics: Dict[str, float] = Field(default_factory=dict)


class ExecutionDecision(BaseModel):
    decision_id: str
    adapter: str
    venue: str
    market_type: str
    created_at: datetime
    requested_by: str
    policy: str
    risk_profile: str
    cost_profile: str
    status: DecisionStatus = "created"


class ExecutionOrder(BaseModel):
    order_id: int
    decision_id: str
    target: str
    track: str
    side: str
    quantity: float
    est_price: Optional[float] = None
    status: ParentOrderStatus = "submitted"
    adapter: str
    venue: str
    time_in_force: str
    strategy_id: str


class ChildOrder(BaseModel):
    child_order_id: Optional[int] = None
    decision_id: str
    parent_order_id: int
    client_order_id: str
    venue_order_id: Optional[str] = None
    symbol: str
    side: str
    qty: float
    limit_price: Optional[float] = None
    tif: str
    status: ChildOrderStatus = "new"
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    slice_index: int = 0


class Fill(BaseModel):
    fill_id: Optional[int] = None
    child_order_id: int
    fill_ts: datetime
    qty: float
    price: float
    fee: float = 0.0
    fee_currency: Optional[str] = None
    liquidity_flag: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    style: Literal["marketable_limit", "passive_twap"]
    child_orders: List[ChildOrder] = Field(default_factory=list)
    market_state_missing: int = 0


__all__ = [
    "ExecutionLifecycleEvent",
    "ExecutionDecision",
    "ExecutionOrder",
    "ChildOrder",
    "Fill",
    "ExecutionPlan",
    "DecisionStatus",
    "ParentOrderStatus",
    "ChildOrderStatus",
]
