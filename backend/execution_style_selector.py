from __future__ import annotations

import os
from typing import Any, Dict, Optional

from risk_state.models import RiskRegime, RiskState


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


class ExecutionStyleSelector:
    def __init__(self) -> None:
        self.exec_deadline_cap_s = _env_int("EXEC_DEADLINE_CAP_S", 60)
        self.twap_slices = _env_int("TWAP_SLICES", 8)
        self.vol_spike_threshold = _env_float("VOL_SPIKE_THRESHOLD", 0.08)
        self.signal_high_threshold = _env_float("SIGNAL_HIGH_THRESHOLD", 0.8)
        self.slippage_high_bps = _env_float("SLIP_YELLOW_BPS", 15.0)
        self.order_adv_threshold = _env_float("ORDER_ADV_THRESHOLD", 0.01)

    def select_style(
        self,
        order_intent: Dict[str, Any],
        risk_state: RiskState,
        market_snapshot: Dict[str, Any],
        exec_stats: Any,
    ) -> Optional[Dict[str, Any]]:
        if str(os.getenv("ENABLE_STYLE_SWITCHING", "1")).strip().lower() not in {"1", "true", "yes", "y", "on"}:
            return {
                "style": "marketable_limit",
                "execution_style": "marketable_limit",
                "deadline_s": int(min(self.exec_deadline_cap_s, max(10, int(0.02 * max(1, int(order_intent.get("horizon_seconds") or 3600)))))),
                "n_slices": 1,
                "twap_slices": 1,
                "slice_interval_s": 1.0,
                "twap_interval_sec": 1.0,
                "chase_policy": 0,
                "max_slippage_bps": float(max(3.0, min(200.0, _env_float("EXEC_MAX_SLIPPAGE_BPS", 20.0)))),
            }
        if risk_state.regime == RiskRegime.RED:
            return None

        score = abs(float(order_intent.get("score", 0.0) or 0.0))
        horizon_seconds = max(1, int(order_intent.get("horizon_seconds") or 3600))
        deadline_s = min(self.exec_deadline_cap_s, max(10, int(0.02 * horizon_seconds)))

        realized_vol = float(market_snapshot.get("realized_vol", 0.0) or 0.0)
        vol_spike = bool(market_snapshot.get("vol_spike")) or realized_vol > self.vol_spike_threshold
        slippage_high = float(getattr(exec_stats, "slippage_bps_p90", 0.0) or 0.0) > self.slippage_high_bps
        adv = float(market_snapshot.get("adv_qty", 0.0) or 0.0)
        order_qty = abs(float(order_intent.get("qty", 0.0) or 0.0))
        order_adv_ratio = (order_qty / adv) if adv > 1e-9 else 0.0
        style_bias = str((risk_state.soft_penalty_factors or {}).get("exec_style_bias", "neutral") or "neutral").lower()

        if vol_spike and score >= self.signal_high_threshold:
            style = "marketable_limit"
            deadline_s = min(deadline_s, 20)
            n_slices = 1
            chase = 1
        elif slippage_high or order_adv_ratio > self.order_adv_threshold or style_bias == "passive":
            style = "passive_twap"
            n_slices = max(2, self.twap_slices)
            deadline_s = max(deadline_s, 20)
            chase = 0
        else:
            style = "marketable_limit"
            n_slices = 1
            chase = 1 if score >= 0.5 else 0

        interval = max(1.0, float(deadline_s) / float(max(1, n_slices)))
        cost_scale = float((risk_state.soft_penalty_factors or {}).get("cost_scale", 1.0) or 1.0)
        max_slippage = max(3.0, min(200.0, _env_float("EXEC_MAX_SLIPPAGE_BPS", 20.0) * cost_scale))
        return {
            "style": style,
            "execution_style": style,
            "deadline_s": int(deadline_s),
            "n_slices": int(n_slices),
            "twap_slices": int(n_slices),
            "slice_interval_s": float(interval),
            "twap_interval_sec": float(interval),
            "chase_policy": int(chase),
            "max_slippage_bps": float(max_slippage),
        }
