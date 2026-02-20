from __future__ import annotations

import json
from typing import Any, Dict, List

from psycopg2.extras import execute_values


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
                    status, adapter, venue, time_in_force, max_slippage_bps, strategy_id, metadata, created_at
                ) VALUES %s
                RETURNING id
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
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
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE orders_sim
                SET status = %s, metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
                WHERE id = %s
                """,
                (status, json.dumps(metadata), order_id),
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
