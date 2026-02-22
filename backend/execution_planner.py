from __future__ import annotations

from typing import Any, Dict, List, Tuple

from execution_models import ChildOrder, ExecutionPlan


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _split_qty(total_qty: float, n_slices: int) -> List[float]:
    n = max(1, int(n_slices))
    if total_qty <= 0:
        return []
    base = total_qty / n
    out = [base for _ in range(n)]
    out[-1] += max(0.0, total_qty - sum(out))
    return out


def _marketable_limit_price(side: str, est_price: float, max_slippage_bps: float) -> float:
    slip = max(0.0, float(max_slippage_bps)) / 10000.0
    if side == "buy":
        return est_price * (1.0 + slip)
    return est_price * (1.0 - slip)


def _passive_limit_price(side: str, est_price: float, spread_bps: float, maker_bias_bps: float) -> float:
    total = max(0.0, spread_bps * 0.5 + maker_bias_bps) / 10000.0
    if side == "buy":
        return est_price * (1.0 - total)
    return est_price * (1.0 + total)


def build_execution_plan(
    *,
    order: Dict[str, Any],
    context: Dict[str, Any],
    market_state: Dict[str, Any],
) -> ExecutionPlan:
    style = str(context.get("execution_style") or "marketable_limit").strip().lower()
    qty = _safe_float(order.get("quantity"), 0.0)
    est_price = _safe_float(order.get("est_price"), 0.0)
    side = str(order.get("side") or "buy").strip().lower()
    parent_order_id = int(order.get("id") or 0)
    decision_id = str(order.get("decision_id") or "")
    symbol = str(order.get("target") or "").upper()
    tif = str(context.get("time_in_force") or "IOC").upper()
    max_slippage_bps = _safe_float(context.get("max_slippage_bps"), 20.0)
    market_state_missing = 0

    spread_bps = _safe_float(market_state.get("spread_bps"), 0.0)
    if spread_bps <= 0 and market_state:
        bid = _safe_float(market_state.get("bid_px"), 0.0)
        ask = _safe_float(market_state.get("ask_px"), 0.0)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        if mid > 0:
            spread_bps = max(0.0, (ask - bid) / mid * 10000.0)
    if est_price <= 0.0:
        bid = _safe_float(market_state.get("bid_px"), 0.0)
        ask = _safe_float(market_state.get("ask_px"), 0.0)
        est_price = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    if est_price <= 0.0:
        market_state_missing = 1
        est_price = 1.0

    child_orders: List[ChildOrder] = []
    if style == "passive_twap":
        n_slices = int(context.get("twap_slices") or 8)
        maker_bias_bps = _safe_float(context.get("maker_bias_bps"), 1.5)
        for i, q in enumerate(_split_qty(qty, n_slices)):
            if q <= 0:
                continue
            limit_px = _passive_limit_price(side, est_price, spread_bps, maker_bias_bps)
            client_order_id = f"{decision_id}:{parent_order_id}:{i}:0"
            child_orders.append(
                ChildOrder(
                    decision_id=decision_id,
                    parent_order_id=parent_order_id,
                    client_order_id=client_order_id,
                    symbol=symbol,
                    side=side,
                    qty=float(round(q, 12)),
                    limit_price=float(round(limit_px, 8)),
                    tif=tif if tif in {"GTC", "IOC", "FOK"} else "GTC",
                    slice_index=i,
                )
            )
    else:
        limit_px = _marketable_limit_price(side, est_price, max_slippage_bps)
        client_order_id = f"{decision_id}:{parent_order_id}:0:0"
        child_orders.append(
            ChildOrder(
                decision_id=decision_id,
                parent_order_id=parent_order_id,
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                qty=float(round(qty, 12)),
                limit_price=float(round(limit_px, 8)),
                tif=tif if tif in {"GTC", "IOC", "FOK"} else "IOC",
                slice_index=0,
            )
        )

    return ExecutionPlan(style=("passive_twap" if style == "passive_twap" else "marketable_limit"), child_orders=child_orders, market_state_missing=market_state_missing)


def deterministic_plan_fingerprint(plan: ExecutionPlan) -> Tuple[str, int]:
    # helper for tests/repro: joined client ids + count
    ids = "|".join(c.client_order_id for c in plan.child_orders)
    return ids, len(plan.child_orders)
