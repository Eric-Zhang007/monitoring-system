from __future__ import annotations

import os
from typing import Any, Dict, List

from account_state.models import AccountState
from risk_state.models import RiskRegime, RiskState


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


class RiskManager:
    def __init__(self) -> None:
        self.margin_yellow = _env_float("MARGIN_YELLOW", 1.3)
        self.margin_red = _env_float("MARGIN_RED", 1.05)
        self.min_free_margin_usd = _env_float("MIN_FREE_MARGIN_USD", 50.0)
        self.slip_yellow_bps = _env_float("SLIP_YELLOW_BPS", 15.0)
        self.reject_red = _env_float("REJECT_RED", 0.60)
        self.vol_spike_threshold = _env_float("VOL_SPIKE_THRESHOLD", 0.08)

    def evaluate(
        self,
        account: AccountState,
        symbol: str,
        order_intent: Dict[str, Any],
        market_snapshot: Dict[str, Any],
    ) -> RiskState:
        _ = (symbol, order_intent)
        hard_enabled = _env_flag("ENABLE_RISK_HARD_LIMITS", "1")
        soft_enabled = _env_flag("ENABLE_SOFT_PENALTIES", "1")
        reasons: List[str] = []
        penalties: Dict[str, Any] = {
            "pos_scale": 1.0,
            "band_scale": 1.0,
            "cost_scale": 1.0,
            "exec_style_bias": "neutral",
        }

        if hard_enabled:
            if not bool(account.health.is_fresh):
                reasons.append("account_state_stale")
            if not bool(account.health.recon_ok):
                reasons.append("reconciliation_not_ok")
            if float(account.balances.margin_ratio) < self.margin_red:
                reasons.append("margin_ratio_red")
            if float(account.balances.free_margin) < self.min_free_margin_usd:
                reasons.append("free_margin_low")
            if float(account.execution_stats.reject_rate_5m) > self.reject_red:
                reasons.append("reject_rate_red")

            if reasons:
                return RiskState(
                    regime=RiskRegime.RED,
                    hard_limits_ok=False,
                    soft_penalty_factors=penalties,
                    reason_codes=reasons,
                )

        if soft_enabled:
            if float(account.execution_stats.slippage_bps_p90) > self.slip_yellow_bps:
                penalties["pos_scale"] *= 0.7
                penalties["cost_scale"] *= 1.3
                penalties["exec_style_bias"] = "passive"
                reasons.append("slippage_p90_high")

            if float(account.balances.margin_ratio) < self.margin_yellow:
                penalties["pos_scale"] *= 0.5
                reasons.append("margin_ratio_yellow")

            realized_vol = float(market_snapshot.get("realized_vol", 0.0) or 0.0)
            if bool(market_snapshot.get("vol_spike")) or realized_vol > self.vol_spike_threshold:
                penalties["band_scale"] *= 1.5
                reasons.append("vol_spike")

        regime = RiskRegime.YELLOW if reasons else RiskRegime.GREEN
        return RiskState(
            regime=regime,
            hard_limits_ok=bool(hard_enabled),
            soft_penalty_factors=penalties,
            reason_codes=reasons,
        )
