from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


def upsert_balances(
    connector,
    *,
    ts: datetime,
    venue: str,
    cash: float,
    equity: float,
    free_margin: float,
    used_margin: float,
    margin_ratio: float,
    account_currency: str = "USD",
    raw: Optional[Dict[str, Any]] = None,
) -> None:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO balances_state (
                    ts, venue, cash, equity, free_margin, used_margin, margin_ratio, account_currency, raw
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (venue, ts)
                DO UPDATE SET
                    cash = EXCLUDED.cash,
                    equity = EXCLUDED.equity,
                    free_margin = EXCLUDED.free_margin,
                    used_margin = EXCLUDED.used_margin,
                    margin_ratio = EXCLUDED.margin_ratio,
                    account_currency = EXCLUDED.account_currency,
                    raw = EXCLUDED.raw
                """,
                (
                    ts,
                    str(venue),
                    float(cash),
                    float(equity),
                    float(free_margin),
                    float(used_margin),
                    float(margin_ratio),
                    str(account_currency or "USD").upper(),
                    json.dumps(raw or {}),
                ),
            )


def upsert_positions(
    connector,
    *,
    ts: datetime,
    venue: str,
    symbol: str,
    qty: float,
    avg_cost: float,
    liq_price: Optional[float] = None,
    unrealized_pnl: float = 0.0,
    realized_pnl: float = 0.0,
    leverage: float = 1.0,
    margin_mode: str = "cross",
    position_mode: str = "one_way",
    raw: Optional[Dict[str, Any]] = None,
) -> None:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO positions_state (
                    ts, venue, symbol, qty, avg_cost, liq_price, unrealized_pnl, realized_pnl,
                    leverage, margin_mode, position_mode, raw
                ) VALUES (%s, %s, UPPER(%s), %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (venue, symbol, ts)
                DO UPDATE SET
                    qty = EXCLUDED.qty,
                    avg_cost = EXCLUDED.avg_cost,
                    liq_price = EXCLUDED.liq_price,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    realized_pnl = EXCLUDED.realized_pnl,
                    leverage = EXCLUDED.leverage,
                    margin_mode = EXCLUDED.margin_mode,
                    position_mode = EXCLUDED.position_mode,
                    raw = EXCLUDED.raw
                """,
                (
                    ts,
                    str(venue),
                    str(symbol),
                    float(qty),
                    float(avg_cost),
                    (float(liq_price) if liq_price is not None else None),
                    float(unrealized_pnl),
                    float(realized_pnl),
                    float(leverage),
                    str(margin_mode),
                    str(position_mode),
                    json.dumps(raw or {}),
                ),
            )


def insert_snapshot(
    connector,
    *,
    ts: datetime,
    venue: str,
    adapter: str,
    equity: float,
    free_margin: float,
    margin_ratio: float,
    raw_state: Dict[str, Any],
) -> int:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO account_state_snapshots (
                    ts, venue, adapter, equity, free_margin, margin_ratio, raw
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                RETURNING id
                """,
                (
                    ts,
                    str(venue),
                    str(adapter),
                    float(equity),
                    float(free_margin),
                    float(margin_ratio),
                    json.dumps(raw_state or {}),
                ),
            )
            row = cur.fetchone() or {}
            return int(row.get("id") or 0)


def insert_decision_trace(connector, trace: Dict[str, Any]) -> int:
    data = dict(trace or {})
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO decision_traces (
                    ts, decision_id, symbol, action, target_pos, delta_pos, exec_style, deadline_s, slices,
                    mu, sigma, direction_prob, cost, risk, account, reason_codes
                ) VALUES (
                    COALESCE(%s, NOW()), %s, UPPER(%s), %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb
                )
                RETURNING id
                """,
                (
                    data.get("ts"),
                    str(data.get("decision_id") or ""),
                    str(data.get("symbol") or ""),
                    str(data.get("action") or "hold"),
                    float(data.get("target_pos") or 0.0),
                    float(data.get("delta_pos") or 0.0),
                    str(data.get("exec_style") or "marketable_limit"),
                    int(data.get("deadline_s") or 0),
                    int(data.get("slices") or 1),
                    json.dumps(data.get("mu") or {}),
                    json.dumps(data.get("sigma") or {}),
                    json.dumps(data.get("direction_prob") or {}),
                    json.dumps(data.get("cost") or {}),
                    json.dumps(data.get("risk") or {}),
                    json.dumps(data.get("account") or {}),
                    json.dumps(list(data.get("reason_codes") or [])),
                ),
            )
            row = cur.fetchone() or {}
            return int(row.get("id") or 0)


def latest_balance(connector, *, venue: str) -> Dict[str, Any]:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM balances_state
                WHERE venue = %s
                ORDER BY ts DESC, id DESC
                LIMIT 1
                """,
                (str(venue),),
            )
            row = cur.fetchone()
            return dict(row) if row else {}


def latest_positions(connector, *, venue: str) -> List[Dict[str, Any]]:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH latest_ts AS (
                    SELECT symbol, MAX(ts) AS ts
                    FROM positions_state
                    WHERE venue = %s
                    GROUP BY symbol
                )
                SELECT p.*
                FROM positions_state p
                JOIN latest_ts t ON t.symbol = p.symbol AND t.ts = p.ts
                WHERE p.venue = %s
                ORDER BY p.symbol ASC
                """,
                (str(venue), str(venue)),
            )
            return [dict(r) for r in cur.fetchall()]


def recent_decision_traces(connector, *, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
    with connector() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM decision_traces
                WHERE symbol = UPPER(%s)
                ORDER BY ts DESC, id DESC
                LIMIT %s
                """,
                (str(symbol), int(limit)),
            )
            return [dict(r) for r in cur.fetchall()]
