#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")
DEFAULT_TOP10 = "BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK"


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit training data completeness for liquid universe")
    ap.add_argument("--database-url", default=DATABASE_URL)
    ap.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", DEFAULT_TOP10))
    ap.add_argument("--lookback-days", type=int, default=180)
    args = ap.parse_args()

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("empty symbols")

    out: Dict[str, Any] = {
        "symbols": symbols,
        "lookback_days": int(args.lookback_days),
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

            expected_1h = max(1, int(args.lookback_days) * 24)
            rows_ready_count = 0
            coverage_values: List[float] = []

            for sym in symbols:
                metrics: Dict[str, Any] = {"symbol": sym}
                if table_flags["market_bars"]:
                    row_1h = _fetch_one(
                        cur,
                        """
                        SELECT
                            COUNT(*)::bigint AS rows_1h,
                            COALESCE(MIN(ts), NOW()) AS min_ts,
                            COALESCE(MAX(ts), NOW()) AS max_ts
                        FROM market_bars
                        WHERE symbol = %s
                          AND timeframe = '1h'
                          AND ts > NOW() - make_interval(days => %s)
                        """,
                        (sym, int(args.lookback_days)),
                    )
                    rows_1h = int(row_1h.get("rows_1h") or 0)
                    coverage_1h = float(rows_1h / max(1, expected_1h))
                    metrics["market_bars_1h_rows"] = rows_1h
                    metrics["market_bars_1h_coverage"] = round(coverage_1h, 4)
                    metrics["market_bars_1h_min_ts"] = row_1h.get("min_ts").isoformat() if row_1h.get("min_ts") else None
                    metrics["market_bars_1h_max_ts"] = row_1h.get("max_ts").isoformat() if row_1h.get("max_ts") else None
                    coverage_values.append(coverage_1h)
                    if rows_1h >= max(200, int(expected_1h * 0.7)):
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
                "symbols_ready_market_bars_1h": int(rows_ready_count),
                "market_bars_1h_ready_ratio": round(float(rows_ready_count / max(1, len(symbols))), 4),
                "market_bars_1h_avg_coverage": round(float(sum(coverage_values) / max(1, len(coverage_values))), 4),
                "training_data_green": bool(rows_ready_count == len(symbols)),
            }

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
