#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import List

import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
DEFAULT_TOP10 = "BTC,ETH,SOL"


def _parse_symbols(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for part in str(raw or "").split(","):
        sym = part.strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def _parse_as_of(raw: str) -> datetime:
    txt = str(raw or "").strip()
    if not txt:
        return datetime.now(timezone.utc)
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser(description="Insert liquid asset universe snapshot for as-of backtest resolution")
    ap.add_argument("--track", default="liquid")
    ap.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", DEFAULT_TOP10))
    ap.add_argument("--as-of", default="")
    ap.add_argument("--version", default="top10_static_v1")
    ap.add_argument("--source", default="manual_seed")
    ap.add_argument("--database-url", default=DATABASE_URL)
    args = ap.parse_args()

    track = str(args.track).strip().lower()
    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("empty symbols")
    as_of = _parse_as_of(args.as_of)

    with psycopg2.connect(args.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO asset_universe_snapshots (
                    track, as_of, universe_version, source, symbols_json, created_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                """,
                (
                    track,
                    as_of,
                    str(args.version).strip() or "manual_v1",
                    str(args.source).strip() or "manual_seed",
                    json.dumps(symbols, ensure_ascii=False),
                ),
            )

    print(
        json.dumps(
            {
                "status": "ok",
                "track": track,
                "as_of": as_of.isoformat(),
                "symbols": symbols,
                "universe_version": str(args.version),
                "source": str(args.source),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
