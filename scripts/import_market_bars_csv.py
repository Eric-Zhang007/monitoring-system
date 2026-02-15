#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import List, Tuple

import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")


def _parse_ts(raw: str) -> datetime:
    txt = str(raw or "").strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    return datetime.fromisoformat(txt)


def _read_rows(path: str) -> List[Tuple]:
    out: List[Tuple] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                (
                    str(row.get("symbol") or "").strip().upper(),
                    str(row.get("timeframe") or "1h").strip().lower(),
                    _parse_ts(str(row.get("ts") or "")),
                    float(row.get("open") or 0.0),
                    float(row.get("high") or 0.0),
                    float(row.get("low") or 0.0),
                    float(row.get("close") or 0.0),
                    float(row.get("volume") or 0.0),
                    int(float(row.get("trades_count") or 0)),
                    str(row.get("source") or "import_csv").strip(),
                )
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Import market_bars csv into PostgreSQL with upsert")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--database-url", default=DATABASE_URL)
    args = ap.parse_args()

    rows = _read_rows(str(args.csv))
    if not rows:
        print(json.dumps({"status": "ok", "inserted": 0}, ensure_ascii=False))
        return 0

    with psycopg2.connect(args.database_url) as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO market_bars (
                    symbol, timeframe, ts, open, high, low, close, volume, trades_count, source
                ) VALUES %s
                ON CONFLICT (symbol, timeframe, ts)
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    trades_count = EXCLUDED.trades_count,
                    source = EXCLUDED.source
                """,
                rows,
                template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                page_size=500,
            )

    print(json.dumps({"status": "ok", "inserted": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
