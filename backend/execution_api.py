from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np


def normalize_reject_reason(reason: str) -> str:
    r = (reason or "").strip().lower()
    if not r:
        return "none"
    if "invalid_quantity" in r:
        return "invalid_quantity"
    if "slippage_too_wide" in r:
        return "slippage_too_wide"
    if "risk_blocked" in r or "kill_switch" in r:
        return "risk_blocked"
    if "venue_http_" in r or "venue_error" in r:
        return "venue_error"
    if "bitget_credentials_not_configured" in r:
        return "bitget_credentials_not_configured"
    if "bitget_signature_error" in r:
        return "bitget_signature_error"
    if "bitget_transport_error" in r:
        return "bitget_transport_error"
    if "bitget_unknown_error" in r:
        return "bitget_unknown_error"
    if "bitget_rate_limited" in r:
        return "bitget_rate_limited"
    if "bitget_symbol_not_supported" in r:
        return "bitget_symbol_not_supported"
    if "bitget_precision_invalid" in r:
        return "bitget_precision_invalid"
    if "bitget_position_rule_violation" in r:
        return "bitget_position_rule_violation"
    if "timeout" in r or "no_fill_after_retries" in r:
        return "timeout_or_no_fill"
    if "paper_reject_simulated" in r:
        return "simulated_reject"
    return "other"


def normalize_execution_payload(execution: Dict[str, Any], now_iso: str) -> Dict[str, Any]:
    out = dict(execution or {})
    lifecycle_in = execution.get("lifecycle") if isinstance(execution, dict) else []
    lifecycle_out: List[Dict[str, Any]] = []
    if isinstance(lifecycle_in, list):
        for ev in lifecycle_in:
            if not isinstance(ev, dict):
                continue
            metrics: Dict[str, float] = {}
            for k in ("filled_qty", "remaining_qty", "avg_fill_price", "retry", "http_code", "timeout_sec"):
                v = ev.get(k)
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
            lifecycle_out.append(
                {
                    "event": str(ev.get("event") or "unknown"),
                    "status": str(ev.get("status") or "unknown"),
                    "time": str(ev.get("time") or now_iso),
                    "metrics": metrics,
                }
            )
    out["lifecycle"] = lifecycle_out
    reason = str(out.get("reject_reason") or "")
    out["reject_reason_category"] = normalize_reject_reason(reason)
    return out


def order_to_risk_weight(
    *,
    order: Dict[str, Any],
    risk_equity_usd: Optional[float],
    signed_notional: Optional[float] = None,
    conversion_error_hook: Optional[Callable[[str], None]] = None,
) -> Optional[float]:
    track = str(order.get("track") or "liquid")
    meta = order.get("metadata") if isinstance(order.get("metadata"), dict) else {}
    if isinstance(meta, dict):
        wd = meta.get("weight_delta")
        if isinstance(wd, (int, float)) and np.isfinite(float(wd)):
            return float(wd)
    if risk_equity_usd is None:
        env_equity = str(os.getenv("RISK_EQUITY_USD", "")).strip()
        if env_equity:
            try:
                risk_equity_usd = float(env_equity)
            except Exception:
                risk_equity_usd = None
    eq = float(risk_equity_usd or 0.0)
    if eq <= 1e-9:
        if conversion_error_hook is not None:
            conversion_error_hook(track)
        return None
    notional = float(signed_notional) if signed_notional is not None else 0.0
    if abs(notional) <= 1e-12:
        qty = float(order.get("quantity") or 0.0)
        if qty <= 0.0:
            if conversion_error_hook is not None:
                conversion_error_hook(track)
            return 0.0
        side = str(order.get("side") or "buy").strip().lower()
        sign = 1.0 if side == "buy" else -1.0
        px = float(order.get("est_price") or 0.0)
        notional = sign * qty * max(0.0, px)
    if abs(notional) <= 1e-12:
        if conversion_error_hook is not None:
            conversion_error_hook(track)
        return 0.0
    return float(notional / eq)


def infer_execution_risk_positions(
    *,
    orders: List[Dict[str, Any]],
    risk_equity_usd: Optional[float],
    order_risk_price_fn: Callable[[Dict[str, Any], Dict[str, float]], float],
    rebalance_position_cls: Type[Any],
    conversion_error_hook: Optional[Callable[[str], None]] = None,
) -> List[Any]:
    price_cache: Dict[str, float] = {}
    explicit_weights: List[Optional[float]] = []
    signed_notionals: List[float] = []
    for o in orders:
        qty = float(o.get("quantity") or 0.0)
        if qty <= 0.0:
            explicit_weights.append(None)
            signed_notionals.append(0.0)
            continue
        side = str(o.get("side") or "buy").lower()
        sign = 1.0 if side == "buy" else -1.0
        signed_qty = sign * qty
        px = order_risk_price_fn(o, price_cache)
        signed_notional = signed_qty * max(0.0, px)
        signed_notionals.append(signed_notional)
        explicit_weights.append(
            order_to_risk_weight(
                order=o,
                risk_equity_usd=risk_equity_usd,
                signed_notional=signed_notional,
                conversion_error_hook=conversion_error_hook,
            )
        )

    total_abs_notional = float(sum(abs(x) for x in signed_notionals))
    inferred: List[Any] = []
    for idx, o in enumerate(orders):
        signed_notional = signed_notionals[idx]
        if abs(signed_notional) <= 1e-12:
            continue
        explicit = explicit_weights[idx]
        if explicit is not None and np.isfinite(float(explicit)):
            weight = float(explicit)
        elif total_abs_notional > 1e-9:
            weight = float(signed_notional / total_abs_notional)
        else:
            weight = 0.0
        inferred.append(
            rebalance_position_cls(
                target=str(o.get("target") or "").upper(),
                track=str(o.get("track") or "liquid"),
                weight=float(weight),
            )
        )
    return inferred
