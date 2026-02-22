from __future__ import annotations

import json
from typing import Any, Dict, List

from psycopg2.extras import execute_values

from repository_modules import execution_oms_store


def create_execution_orders(
    connector,
    *,
    decision_id: str,
    adapter: str,
    venue: str,
    time_in_force: str,
    max_slippage_bps: float,
    orders: List[Dict[str, Any]],
) -> List[int]:
    if not orders:
        return []
    first = orders[0]
    first_meta = first.get("metadata") if isinstance(first.get("metadata"), dict) else {}
    exec_params = first_meta.get("execution_params") if isinstance(first_meta.get("execution_params"), dict) else {}
    execution_oms_store.create_decision(
        connector,
        {
            "decision_id": decision_id,
            "adapter": adapter,
            "venue": venue,
            "market_type": str(exec_params.get("market_type") or "spot"),
            "product_type": exec_params.get("product_type"),
            "leverage": exec_params.get("leverage"),
            "reduce_only": bool(exec_params.get("reduce_only") or False),
            "position_mode": exec_params.get("position_mode"),
            "margin_mode": exec_params.get("margin_mode"),
            "requested_by": "api",
            "strategy_id": str(first.get("strategy_id") or "default-liquid-v1"),
            "policy_snapshot": {
                "submit_time_in_force": str(time_in_force),
                "submit_max_slippage_bps": float(max_slippage_bps),
            },
            "status": "created",
        },
    )
    with connector() as conn:
        with conn.cursor() as cur:
            rows = [
                (
                    decision_id,
                    o["target"],
                    o.get("track", "liquid"),
                    o["side"],
                    o.get("quantity", 0.0),
                    o.get("est_price"),
                    float(max_slippage_bps),
                    "submitted",
                    adapter,
                    venue,
                    time_in_force,
                    float(max_slippage_bps),
                    o.get("strategy_id", "default-liquid-v1"),
                    json.dumps(o.get("metadata", {})),
                )
                for o in orders
            ]
            inserted = execute_values(
                cur,
                """
                INSERT INTO orders_sim (
                    decision_id, target, track, side, quantity, est_price, est_cost_bps,
                    status, adapter, venue, time_in_force, max_slippage_bps, strategy_id, metadata, created_at, updated_at
                ) VALUES %s
                RETURNING id
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())",
                fetch=True,
            )
            return [int(r["id"]) for r in inserted]


def fetch_orders_for_decision(connector, decision_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM orders_sim
                WHERE decision_id = %s
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (decision_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]


def update_order_execution(connector, order_id: int, status: str, metadata: Dict[str, Any]) -> None:
    execution = metadata.get("execution") if isinstance(metadata.get("execution"), dict) else {}
    filled_qty = float(execution.get("filled_qty") or metadata.get("filled_qty") or 0.0)
    avg_fill_price = execution.get("avg_fill_price")
    fees_paid = float(execution.get("fees_paid") or metadata.get("fees_paid") or 0.0)
    last_venue_order_id = execution.get("venue_order_id") or metadata.get("venue_order_id")
    reject_reason = execution.get("reject_reason") or metadata.get("reject_reason")
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE orders_sim
                SET status = %s,
                    filled_qty = %s,
                    avg_fill_price = %s,
                    fees_paid = %s,
                    last_venue_order_id = %s,
                    reject_reason = %s,
                    metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (
                    status,
                    filled_qty,
                    avg_fill_price,
                    fees_paid,
                    last_venue_order_id,
                    reject_reason,
                    json.dumps(metadata),
                    order_id,
                ),
            )


def save_risk_event(
    connector,
    *,
    decision_id: str,
    severity: str,
    code: str,
    message: str,
    payload: Dict[str, Any],
) -> None:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO risk_events (
                    decision_id, severity, code, message, payload, created_at
                ) VALUES (%s, %s, %s, %s, %s, NOW())
                """,
                (decision_id, severity, code, message, json.dumps(payload)),
            )
