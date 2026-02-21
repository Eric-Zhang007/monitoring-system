from __future__ import annotations

import os
from typing import Any, Dict


SHORT_HORIZONS = {"1h", "4h"}
LONG_HORIZONS = {"1d", "7d"}


def _safe_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _horizon_from_order(order: Dict[str, Any]) -> str:
    meta = order.get("metadata") if isinstance(order.get("metadata"), dict) else {}
    horizon = str(meta.get("horizon") or meta.get("selected_horizon") or meta.get("horizon_bucket") or "").strip().lower()
    if horizon in SHORT_HORIZONS | LONG_HORIZONS:
        return horizon
    return "1h"


def resolve_order_execution_context(order: Dict[str, Any], base_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    base = dict(base_context or {})
    horizon = _horizon_from_order(order)
    if horizon in SHORT_HORIZONS:
        policy_name = "short_horizon_exec"
        cfg = {
            "time_in_force": os.getenv("EXEC_SHORT_TIF", "IOC"),
            "max_slippage_bps": _safe_float(os.getenv("EXEC_SHORT_MAX_SLIPPAGE_BPS", "18"), 18.0),
            "limit_timeout_sec": _safe_float(os.getenv("EXEC_SHORT_LIMIT_TIMEOUT_SEC", "1.5"), 1.5),
            "max_retries": _safe_int(os.getenv("EXEC_SHORT_MAX_RETRIES", "1"), 1),
            "execution_style": "marketable_limit",
            "max_participation_rate": _safe_float(os.getenv("EXEC_SHORT_MAX_PARTICIPATION", "0.35"), 0.35),
        }
    else:
        policy_name = "long_horizon_exec"
        cfg = {
            "time_in_force": os.getenv("EXEC_LONG_TIF", "GTC"),
            "max_slippage_bps": _safe_float(os.getenv("EXEC_LONG_MAX_SLIPPAGE_BPS", "10"), 10.0),
            "limit_timeout_sec": _safe_float(os.getenv("EXEC_LONG_LIMIT_TIMEOUT_SEC", "4.0"), 4.0),
            "max_retries": _safe_int(os.getenv("EXEC_LONG_MAX_RETRIES", "3"), 3),
            "execution_style": "passive_twap",
            "max_participation_rate": _safe_float(os.getenv("EXEC_LONG_MAX_PARTICIPATION", "0.12"), 0.12),
        }
    out = {
        **base,
        **cfg,
        "execution_policy": policy_name,
        "horizon": horizon,
    }
    return out


def build_execution_trace(order: Dict[str, Any], execution: Dict[str, Any]) -> Dict[str, Any]:
    est_price = _safe_float(order.get("est_price"), 0.0)
    side = str(order.get("side") or "buy").strip().lower()
    filled_qty = _safe_float(execution.get("filled_qty"), 0.0)
    fill_price = execution.get("avg_fill_price")
    fill_price_f = _safe_float(fill_price, 0.0) if fill_price is not None else 0.0
    fee = _safe_float(execution.get("fees_paid"), 0.0)
    slip_bps = 0.0
    if est_price > 0 and fill_price_f > 0:
        if side == "buy":
            slip_bps = (fill_price_f - est_price) / est_price * 10000.0
        else:
            slip_bps = (est_price - fill_price_f) / est_price * 10000.0
    impact = max(0.0, abs(slip_bps) - _safe_float(os.getenv("EXEC_SLIPPAGE_BASELINE_BPS", "3.0"), 3.0))
    return {
        "theoretical_price": float(est_price),
        "avg_fill_price": float(fill_price_f) if fill_price is not None else None,
        "filled_qty": float(filled_qty),
        "slippage_bps": float(slip_bps),
        "fees_paid": float(fee),
        "impact_bps": float(impact),
    }
