from __future__ import annotations

from typing import Any, Dict, List


def apply_fills_to_position(prev_pos: Dict[str, Any], fills: List[Dict[str, Any]], side: str) -> Dict[str, Any]:
    qty_prev = float(prev_pos.get("position_qty") or 0.0)
    avg_prev = float(prev_pos.get("avg_cost") or 0.0)

    signed = 1.0 if str(side).lower() == "buy" else -1.0
    qty_delta = sum(signed * float(f.get("qty") or 0.0) for f in fills)
    notional_delta = sum(signed * float(f.get("qty") or 0.0) * float(f.get("price") or 0.0) for f in fills)

    qty_new = qty_prev + qty_delta
    if abs(qty_new) <= 1e-12:
        return {"position_qty": 0.0, "avg_cost": 0.0}

    # Same direction update keeps weighted average; direction flip resets basis from current fill mix.
    if qty_prev == 0 or (qty_prev > 0 and qty_new > 0 and qty_delta > 0) or (qty_prev < 0 and qty_new < 0 and qty_delta < 0):
        notional_prev = qty_prev * avg_prev
        avg_new = (notional_prev + notional_delta) / qty_new
    else:
        fill_qty_abs = sum(abs(float(f.get("qty") or 0.0)) for f in fills)
        fill_notional_abs = sum(abs(float(f.get("qty") or 0.0) * float(f.get("price") or 0.0)) for f in fills)
        avg_fill = fill_notional_abs / max(fill_qty_abs, 1e-12)
        avg_new = avg_fill

    return {"position_qty": float(qty_new), "avg_cost": float(avg_new)}


def compute_realized_pnl(prev_pos: Dict[str, Any], fills: List[Dict[str, Any]], side: str) -> float:
    qty_prev = float(prev_pos.get("position_qty") or 0.0)
    avg_prev = float(prev_pos.get("avg_cost") or 0.0)
    if abs(qty_prev) <= 1e-12:
        return 0.0

    sign_prev = 1.0 if qty_prev > 0 else -1.0
    sign_trade = 1.0 if str(side).lower() == "buy" else -1.0
    if sign_prev == sign_trade:
        return 0.0

    close_qty = min(abs(qty_prev), sum(abs(float(f.get("qty") or 0.0)) for f in fills))
    if close_qty <= 0:
        return 0.0
    avg_fill = sum(abs(float(f.get("qty") or 0.0)) * float(f.get("price") or 0.0) for f in fills) / max(sum(abs(float(f.get("qty") or 0.0)) for f in fills), 1e-12)
    if sign_prev > 0:
        return float((avg_fill - avg_prev) * close_qty)
    return float((avg_prev - avg_fill) * close_qty)


def compute_unrealized_pnl(pos: Dict[str, Any], mark_price: float) -> float:
    qty = float(pos.get("position_qty") or 0.0)
    avg = float(pos.get("avg_cost") or 0.0)
    px = float(mark_price or 0.0)
    if abs(qty) <= 1e-12 or px <= 0:
        return 0.0
    return float((px - avg) * qty)
