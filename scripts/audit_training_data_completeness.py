#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
DEFAULT_TOP10 = "BTC,ETH,SOL"


def _parse_symbols(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in str(raw or "").split(","):
        sym = item.strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def _table_exists(cur, table: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s
        ) AS exists_flag
        """,
        (table,),
    )
    row = cur.fetchone() or {}
    return bool(row.get("exists_flag"))


def _fetch_one(cur, sql: str, params: tuple[Any, ...]) -> Dict[str, Any]:
    cur.execute(sql, params)
    return dict(cur.fetchone() or {})


def _timeframe_minutes(timeframe: str) -> int:
    tf = str(timeframe or "1h").strip().lower()
    try:
        if tf.endswith("m"):
            return max(1, int(tf[:-1] or "1"))
        if tf.endswith("h"):
            return max(1, int(tf[:-1] or "1")) * 60
        if tf.endswith("d"):
            return max(1, int(tf[:-1] or "1")) * 1440
        return max(1, int(tf))
    except Exception:
        return 60


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit training data completeness for liquid universe")
    ap.add_argument("--database-url", default=DATABASE_URL)
    ap.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", DEFAULT_TOP10))
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--timeframe", default=os.getenv("LIQUID_PRIMARY_TIMEFRAME", "1h"))
    args = ap.parse_args()

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("empty symbols")
    timeframe = str(args.timeframe).strip().lower() or "1h"
    tf_minutes = _timeframe_minutes(timeframe)

    out: Dict[str, Any] = {
        "symbols": symbols,
        "lookback_days": int(args.lookback_days),
        "timeframe": timeframe,
        "tables": {},
        "per_symbol": {},
        "summary": {},
    }

    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            table_flags = {
                "market_bars": _table_exists(cur, "market_bars"),
                "prices": _table_exists(cur, "prices"),
                "orderbook_l2": _table_exists(cur, "orderbook_l2"),
                "funding_rates": _table_exists(cur, "funding_rates"),
                "onchain_signals": _table_exists(cur, "onchain_signals"),
                "events": _table_exists(cur, "events"),
                "event_links": _table_exists(cur, "event_links"),
                "entities": _table_exists(cur, "entities"),
            }
            out["tables"] = table_flags

            expected_rows = max(1, int((int(args.lookback_days) * 24 * 60) / max(1, tf_minutes)))
            rows_ready_count = 0
            coverage_values: List[float] = []

            for sym in symbols:
                metrics: Dict[str, Any] = {"symbol": sym}
                if table_flags["market_bars"]:
                    row_tf = _fetch_one(
                        cur,
                        """
                        SELECT
                            COUNT(*)::bigint AS rows_tf,
                            COALESCE(MIN(ts), NOW()) AS min_ts,
                            COALESCE(MAX(ts), NOW()) AS max_ts
                        FROM market_bars
                        WHERE symbol = %s
                          AND timeframe = %s
                          AND ts > NOW() - make_interval(days => %s)
                        """,
                        (sym, timeframe, int(args.lookback_days)),
                    )
                    rows_tf = int(row_tf.get("rows_tf") or 0)
                    coverage_tf = float(rows_tf / max(1, expected_rows))
                    metrics["market_bars_rows"] = rows_tf
                    metrics["market_bars_coverage"] = round(coverage_tf, 4)
                    metrics["market_bars_min_ts"] = row_tf.get("min_ts").isoformat() if row_tf.get("min_ts") else None
                    metrics["market_bars_max_ts"] = row_tf.get("max_ts").isoformat() if row_tf.get("max_ts") else None
                    # backward-compatible keys for existing automation
                    if timeframe == "1h":
                        metrics["market_bars_1h_rows"] = rows_tf
                        metrics["market_bars_1h_coverage"] = round(coverage_tf, 4)
                        metrics["market_bars_1h_min_ts"] = metrics["market_bars_min_ts"]
                        metrics["market_bars_1h_max_ts"] = metrics["market_bars_max_ts"]
                    coverage_values.append(coverage_tf)
                    if rows_tf >= max(200, int(expected_rows * 0.7)):
                        rows_ready_count += 1
                if table_flags["funding_rates"]:
                    fr = _fetch_one(
                        cur,
                        """
                        SELECT COUNT(*)::bigint AS rows_cnt
                        FROM funding_rates
                        WHERE symbol = %s
                          AND ts > NOW() - make_interval(days => %s)
                        """,
                        (sym, int(args.lookback_days)),
                    )
                    metrics["funding_rows"] = int(fr.get("rows_cnt") or 0)
                if table_flags["orderbook_l2"]:
                    ob = _fetch_one(
                        cur,
                        """
                        SELECT COUNT(*)::bigint AS rows_cnt
                        FROM orderbook_l2
                        WHERE symbol = %s
                          AND ts > NOW() - make_interval(days => %s)
                        """,
                        (sym, int(args.lookback_days)),
                    )
                    metrics["orderbook_rows"] = int(ob.get("rows_cnt") or 0)
                if table_flags["onchain_signals"]:
                    oc = _fetch_one(
                        cur,
                        """
                        SELECT COUNT(*)::bigint AS rows_cnt
                        FROM onchain_signals
                        WHERE asset_symbol = %s
                          AND ts > NOW() - make_interval(days => %s)
                        """,
                        (sym, int(args.lookback_days)),
                    )
                    metrics["onchain_rows"] = int(oc.get("rows_cnt") or 0)

                if table_flags["events"] and table_flags["event_links"] and table_flags["entities"]:
                    ev = _fetch_one(
                        cur,
                        """
                        SELECT COUNT(*)::bigint AS rows_cnt
                        FROM events e
                        JOIN event_links el ON el.event_id = e.id
                        JOIN entities en ON en.id = el.entity_id
                        WHERE UPPER(en.symbol) = %s
                          AND COALESCE(e.available_at, e.occurred_at) > NOW() - make_interval(days => %s)
                        """,
                        (sym, int(args.lookback_days)),
                    )
                    metrics["linked_event_rows"] = int(ev.get("rows_cnt") or 0)

                out["per_symbol"][sym] = metrics

            out["summary"] = {
                "symbols_count": len(symbols),
                "symbols_ready_market_bars": int(rows_ready_count),
                "market_bars_ready_ratio": round(float(rows_ready_count / max(1, len(symbols))), 4),
                "market_bars_avg_coverage": round(float(sum(coverage_values) / max(1, len(coverage_values))), 4),
                "expected_rows_per_symbol": int(expected_rows),
                "training_data_green": bool(rows_ready_count == len(symbols)),
            }
            if timeframe == "1h":
                out["summary"]["symbols_ready_market_bars_1h"] = int(rows_ready_count)
                out["summary"]["market_bars_1h_ready_ratio"] = out["summary"]["market_bars_ready_ratio"]
                out["summary"]["market_bars_1h_avg_coverage"] = out["summary"]["market_bars_avg_coverage"]

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
