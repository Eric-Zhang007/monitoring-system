from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BalanceState(BaseModel):
    cash: float = 0.0
    equity: float = 0.0
    free_margin: float = 0.0
    used_margin: float = 0.0
    margin_ratio: float = 0.0
    account_currency: str = "USD"


class PositionState(BaseModel):
    symbol: str
    qty: float = 0.0
    avg_cost: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: float = 1.0
    liq_price: float | None = None
    margin_mode: str = "cross"
    position_mode: str = "one_way"


class OrderState(BaseModel):
    client_order_id: str = ""
    venue_order_id: str = ""
    symbol: str = ""
    side: str = "buy"
    qty: float = 0.0
    filled_qty: float = 0.0
    status: str = "submitted"
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ExecutionStats(BaseModel):
    slippage_bps_p50: float = 0.0
    slippage_bps_p90: float = 0.0
    reject_rate_5m: float = 0.0
    fill_latency_p50: float = 0.0
    last_fill_ts: datetime | None = None


class AccountHealth(BaseModel):
    is_fresh: bool = False
    recon_ok: bool = False
    ws_ok: bool = False
    last_error: str = ""


class AccountState(BaseModel):
    ts: datetime = Field(default_factory=_utcnow)
    venue: str = "coinbase"
    adapter: str = "paper"
    balances: BalanceState = Field(default_factory=BalanceState)
    positions: Dict[str, PositionState] = Field(default_factory=dict)
    open_orders: List[OrderState] = Field(default_factory=list)
    execution_stats: ExecutionStats = Field(default_factory=ExecutionStats)
    health: AccountHealth = Field(default_factory=AccountHealth)
    raw: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "venue": self.venue,
            "adapter": self.adapter,
            "equity": float(self.balances.equity),
            "free_margin": float(self.balances.free_margin),
            "margin_ratio": float(self.balances.margin_ratio),
            "open_orders": int(len(self.open_orders)),
            "position_count": int(len(self.positions)),
            "health": self.health.model_dump(),
        }
