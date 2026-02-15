#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")


def _default_map() -> Dict[str, List[str]]:
    return {
        "BTC": ["BTC", "BTC_BG2025_PERP", "BTC_BG2025_SPOT", "BTC_BG2025"],
        "ETH": ["ETH", "ETH_BG2025_PERP", "ETH_BG2025_SPOT", "ETH_BG2025"],
        "SOL": ["SOL", "SOL_BG2025_PERP", "SOL_BG2025_SPOT", "SOL_BG2025"],
    }


def _build_mapping_rows(symbol_map: Dict[str, Sequence[str]], targets: Sequence[str]) -> List[Tuple[str, str, int]]:
    allow = {t.strip().upper() for t in targets if t.strip()}
    out: List[Tuple[str, str, int]] = []
    for target, sources in symbol_map.items():
        tgt = target.strip().upper()
        if allow and tgt not in allow:
            continue
        for idx, src in enumerate(sources):
            src_norm = src.strip().upper()
            if src_norm:
                out.append((tgt, src_norm, idx + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill market_bars from prices with symbol alias mapping")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start", default="2025-01-01T00:00:00+00:00")
    ap.add_argument("--end", default=datetime.now(timezone.utc).isoformat())
    ap.add_argument("--targets", default="BTC,ETH,SOL")
    ap.add_argument("--replace-window", action="store_true")
    ap.add_argument("--symbol-map-json", default="")
    args = ap.parse_args()

    symbol_map = _default_map()
    if args.symbol_map_json.strip():
        symbol_map = json.loads(args.symbol_map_json)
    targets = [s.strip().upper() for s in str(args.targets).split(",") if s.strip()]
    mapping_rows = _build_mapping_rows(symbol_map, targets)
    if not mapping_rows:
        raise SystemExit("empty mapping rows")

    with psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            if args.replace_window:
                cur.execute(
                    """
                    DELETE FROM market_bars
                    WHERE timeframe = %s
                      AND symbol = ANY(%s)
                      AND ts >= %s::timestamptz
                      AND ts <= %s::timestamptz
                    """,
                    (args.timeframe, targets, args.start, args.end),
                )

            values_sql = ",".join(["(%s,%s,%s)"] * len(mapping_rows))
            params: List[object] = []
            for row in mapping_rows:
                params.extend(row)
            params.extend([args.timeframe, args.start, args.end])
            upsert_sql = f"""
            WITH mapping(target, src, priority) AS (
              VALUES {values_sql}
            ),
            src_rows AS (
              SELECT
                m.target AS symbol,
                %s::text AS timeframe,
                p.timestamp AS ts,
                p.price::double precision AS close,
                COALESCE(p.volume::double precision, 0.0) AS volume,
                m.priority
              FROM prices p
              JOIN mapping m ON UPPER(p.symbol) = m.src
              WHERE p.timestamp >= %s::timestamptz
                AND p.timestamp <= %s::timestamptz
            ),
            ranked AS (
              SELECT
                symbol, timeframe, ts, close, volume,
                ROW_NUMBER() OVER (
                  PARTITION BY symbol, timeframe, ts
                  ORDER BY priority ASC
                ) AS rn
              FROM src_rows
            ),
            chosen AS (
              SELECT symbol, timeframe, ts, close, volume
              FROM ranked
              WHERE rn = 1
            )
            INSERT INTO market_bars (
              symbol, timeframe, ts, open, high, low, close, volume, trades_count, source, created_at
            )
            SELECT
              symbol, timeframe, ts, close, close, close, close, volume, 0, 'prices_backfill', NOW()
            FROM chosen
            ON CONFLICT (symbol, timeframe, ts)
            DO UPDATE SET
              open = EXCLUDED.open,
              high = EXCLUDED.high,
              low = EXCLUDED.low,
              close = EXCLUDED.close,
              volume = EXCLUDED.volume,
              source = EXCLUDED.source
            """
            cur.execute(upsert_sql, params)
            affected = int(cur.rowcount or 0)

            cur.execute(
                """
                SELECT symbol, COUNT(*)::bigint AS rows, MIN(ts) AS min_ts, MAX(ts) AS max_ts
                FROM market_bars
                WHERE timeframe = %s AND symbol = ANY(%s)
                GROUP BY symbol
                ORDER BY symbol
                """,
                (args.timeframe, targets),
            )
            stats = [dict(r) for r in cur.fetchall()]

    out = {
        "status": "ok",
        "timeframe": args.timeframe,
        "window": {"start": args.start, "end": args.end},
        "targets": targets,
        "replace_window": bool(args.replace_window),
        "mapping_rows": len(mapping_rows),
        "affected_rows": affected,
        "stats": stats,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
