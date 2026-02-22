from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from psycopg2.extras import execute_values


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_decision(connector, decision_payload: Dict[str, Any]) -> str:
    decision_id = str(decision_payload.get("decision_id") or "").strip()
    if not decision_id:
        raise ValueError("decision_id_required")
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO execution_decisions (
                    decision_id, adapter, venue, market_type, product_type, leverage,
                    reduce_only, position_mode, margin_mode, requested_by, strategy_id,
                    policy_snapshot, risk_snapshot, trace_summary, status, error,
                    created_at, started_at, ended_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s::jsonb, %s, %s,
                    COALESCE(%s, NOW()), %s, %s
                )
                ON CONFLICT (decision_id)
                DO UPDATE SET
                    adapter = EXCLUDED.adapter,
                    venue = EXCLUDED.venue,
                    market_type = EXCLUDED.market_type,
                    product_type = EXCLUDED.product_type,
                    leverage = EXCLUDED.leverage,
                    reduce_only = EXCLUDED.reduce_only,
                    position_mode = EXCLUDED.position_mode,
                    margin_mode = EXCLUDED.margin_mode,
                    requested_by = EXCLUDED.requested_by,
                    strategy_id = EXCLUDED.strategy_id,
                    policy_snapshot = EXCLUDED.policy_snapshot,
                    risk_snapshot = EXCLUDED.risk_snapshot,
                    trace_summary = EXCLUDED.trace_summary,
                    status = EXCLUDED.status,
                    error = EXCLUDED.error
                """,
                (
                    decision_id,
                    str(decision_payload.get("adapter") or "paper"),
                    str(decision_payload.get("venue") or "coinbase"),
                    str(decision_payload.get("market_type") or "spot"),
                    decision_payload.get("product_type"),
                    decision_payload.get("leverage"),
                    bool(decision_payload.get("reduce_only") or False),
                    decision_payload.get("position_mode"),
                    decision_payload.get("margin_mode"),
                    str(decision_payload.get("requested_by") or "api"),
                    str(decision_payload.get("strategy_id") or "default-liquid-v1"),
                    json.dumps(decision_payload.get("policy_snapshot") or {}),
                    json.dumps(decision_payload.get("risk_snapshot") or {}),
                    json.dumps(decision_payload.get("trace_summary") or {}),
                    str(decision_payload.get("status") or "created"),
                    decision_payload.get("error"),
                    decision_payload.get("created_at"),
                    decision_payload.get("started_at"),
                    decision_payload.get("ended_at"),
                ),
            )
    return decision_id


def start_decision(connector, decision_id: str) -> None:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE execution_decisions
                SET status = 'running', started_at = COALESCE(started_at, NOW())
                WHERE decision_id = %s
                """,
                (str(decision_id),),
            )


def finish_decision(connector, decision_id: str, status: str, summary: Dict[str, Any], error: Optional[str] = None) -> None:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE execution_decisions
                SET status = %s,
                    trace_summary = COALESCE(trace_summary, '{}'::jsonb) || %s::jsonb,
                    error = %s,
                    ended_at = NOW()
                WHERE decision_id = %s
                """,
                (str(status), json.dumps(summary or {}), error, str(decision_id)),
            )


def create_child_orders(connector, parent_order_id: int, child_rows: Sequence[Dict[str, Any]]) -> List[int]:
    if not child_rows:
        return []
    with connector() as conn:
        with conn.cursor() as cur:
            rows = [
                (
                    str(r.get("decision_id") or ""),
                    int(parent_order_id),
                    str(r.get("client_order_id") or ""),
                    r.get("venue_order_id"),
                    str(r.get("symbol") or ""),
                    str(r.get("side") or ""),
                    float(r.get("qty") or 0.0),
                    r.get("limit_price"),
                    str(r.get("tif") or "IOC"),
                    str(r.get("status") or "new"),
                    int(r.get("slice_index") or 0),
                    json.dumps(r.get("lifecycle") or []),
                    r.get("submitted_at"),
                    r.get("updated_at"),
                )
                for r in child_rows
            ]
            inserted = execute_values(
                cur,
                """
                INSERT INTO execution_child_orders (
                    decision_id, parent_order_id, client_order_id, venue_order_id,
                    symbol, side, qty, limit_price, tif, status, slice_index,
                    lifecycle, submitted_at, updated_at
                ) VALUES %s
                ON CONFLICT (client_order_id)
                DO UPDATE SET
                    venue_order_id = COALESCE(EXCLUDED.venue_order_id, execution_child_orders.venue_order_id),
                    status = EXCLUDED.status,
                    lifecycle = CASE
                        WHEN jsonb_typeof(EXCLUDED.lifecycle) = 'array' THEN EXCLUDED.lifecycle
                        ELSE execution_child_orders.lifecycle
                    END,
                    submitted_at = COALESCE(EXCLUDED.submitted_at, execution_child_orders.submitted_at),
                    updated_at = COALESCE(EXCLUDED.updated_at, NOW())
                RETURNING id
                """,
                rows,
                template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s)",
                fetch=True,
            )
            return [int(r["id"]) for r in inserted]


def update_child_status(
    connector,
    child_id: int,
    status: str,
    venue_order_id: Optional[str] = None,
    lifecycle_append: Optional[Dict[str, Any]] = None,
) -> None:
    with connector() as conn:
        with conn.cursor() as cur:
            if lifecycle_append:
                cur.execute(
                    """
                    UPDATE execution_child_orders
                    SET status = %s,
                        venue_order_id = COALESCE(%s, venue_order_id),
                        lifecycle = COALESCE(lifecycle, '[]'::jsonb) || %s::jsonb,
                        submitted_at = CASE WHEN status='new' AND %s IN ('submitted','filled','partially_filled') THEN COALESCE(submitted_at, NOW()) ELSE submitted_at END,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (str(status), venue_order_id, json.dumps([lifecycle_append]), str(status), int(child_id)),
                )
            else:
                cur.execute(
                    """
                    UPDATE execution_child_orders
                    SET status = %s,
                        venue_order_id = COALESCE(%s, venue_order_id),
                        submitted_at = CASE WHEN status='new' AND %s IN ('submitted','filled','partially_filled') THEN COALESCE(submitted_at, NOW()) ELSE submitted_at END,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (str(status), venue_order_id, str(status), int(child_id)),
                )


def insert_fills(connector, child_id: int, fills: Sequence[Dict[str, Any]]) -> List[int]:
    if not fills:
        return []
    with connector() as conn:
        with conn.cursor() as cur:
            rows = [
                (
                    int(child_id),
                    f.get("fill_ts") or _utcnow(),
                    float(f.get("qty") or 0.0),
                    float(f.get("price") or 0.0),
                    float(f.get("fee") or 0.0),
                    f.get("fee_currency"),
                    f.get("liquidity_flag"),
                    json.dumps(f.get("raw") or {}),
                )
                for f in fills
                if float(f.get("qty") or 0.0) > 0 and float(f.get("price") or 0.0) > 0
            ]
            if not rows:
                return []
            inserted = execute_values(
                cur,
                """
                INSERT INTO execution_fills (
                    child_order_id, fill_ts, qty, price, fee, fee_currency, liquidity_flag, raw
                ) VALUES %s
                RETURNING id
                """,
                rows,
                template="(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)",
                fetch=True,
            )
            return [int(r["id"]) for r in inserted]


def update_parent_from_fills(connector, parent_order_id: int) -> Dict[str, Any]:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COALESCE(SUM(f.qty), 0.0) AS filled_qty,
                    COALESCE(SUM(f.qty * f.price), 0.0) AS notional,
                    COALESCE(SUM(f.fee), 0.0) AS fees,
                    MAX(c.venue_order_id) AS last_venue_order_id
                FROM execution_child_orders c
                LEFT JOIN execution_fills f ON f.child_order_id = c.id
                WHERE c.parent_order_id = %s
                """,
                (int(parent_order_id),),
            )
            agg = dict(cur.fetchone() or {})

            cur.execute("SELECT quantity FROM orders_sim WHERE id = %s", (int(parent_order_id),))
            row = cur.fetchone() or {}
            qty = float(row.get("quantity") or 0.0)

            filled_qty = float(agg.get("filled_qty") or 0.0)
            notional = float(agg.get("notional") or 0.0)
            fees = float(agg.get("fees") or 0.0)
            avg = notional / filled_qty if filled_qty > 1e-12 else None

            if filled_qty <= 1e-12:
                status = "submitted"
            elif filled_qty + 1e-12 < qty:
                status = "partially_filled"
            else:
                status = "filled"

            cur.execute(
                """
                UPDATE orders_sim
                SET status = %s,
                    filled_qty = %s,
                    avg_fill_price = %s,
                    fees_paid = %s,
                    last_venue_order_id = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (status, filled_qty, avg, fees, agg.get("last_venue_order_id"), int(parent_order_id)),
            )

            return {
                "status": status,
                "filled_qty": filled_qty,
                "avg_fill_price": avg,
                "fees_paid": fees,
                "last_venue_order_id": agg.get("last_venue_order_id"),
            }


def append_reconciliation_log(
    connector,
    *,
    venue: str,
    adapter: str,
    decision_id: Optional[str],
    open_orders_diff: Dict[str, Any],
    positions_diff: Dict[str, Any],
    actions_taken: List[Dict[str, Any]],
    status: str,
    error: Optional[str] = None,
) -> int:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reconciliation_logs (
                    venue, adapter, decision_id, checked_at,
                    open_orders_diff, positions_diff, actions_taken, status, error
                ) VALUES (
                    %s, %s, %s, NOW(),
                    %s::jsonb, %s::jsonb, %s::jsonb, %s, %s
                ) RETURNING id
                """,
                (
                    str(venue),
                    str(adapter),
                    decision_id,
                    json.dumps(open_orders_diff or {}),
                    json.dumps(positions_diff or {}),
                    json.dumps(actions_taken or []),
                    str(status),
                    error,
                ),
            )
            row = cur.fetchone() or {}
            return int(row.get("id") or 0)
