#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import List

import psycopg2


def _get_symbols() -> List[str]:
    raw = os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK")
    out = []
    for part in raw.split(","):
        sym = part.strip().upper()
        if sym:
            out.append(sym)
    return out or ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "TRX", "AVAX", "LINK"]


def main() -> None:
    db_url = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")
    symbols = _get_symbols()
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            entity_ids = {}
            for sym in symbols:
                cur.execute(
                    """
                    INSERT INTO entities (entity_type, name, symbol, country, sector, metadata, created_at, updated_at)
                    VALUES ('asset', %s, %s, NULL, 'crypto', '{}'::jsonb, NOW(), NOW())
                    ON CONFLICT (entity_type, name)
                    DO UPDATE SET symbol = EXCLUDED.symbol, updated_at = NOW()
                    RETURNING id
                    """,
                    (sym, sym),
                )
                entity_ids[sym] = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT id
                FROM events
                WHERE created_at >= NOW() - INTERVAL '7 days'
                  AND (
                    COALESCE(payload->>'market_scope', '') = 'crypto'
                    OR source_name = 'redis-news'
                  )
                """
            )
            event_ids = [int(r[0]) for r in cur.fetchall()]

            inserted = 0
            for event_id in event_ids:
                for sym in symbols:
                    cur.execute(
                        """
                        INSERT INTO event_links (event_id, entity_id, role, created_at)
                        VALUES (%s, %s, 'mentioned', NOW())
                        ON CONFLICT (event_id, entity_id, role) DO NOTHING
                        """,
                        (event_id, entity_ids[sym]),
                    )
                    inserted += cur.rowcount

            conn.commit()
            print(
                {
                    "events_scanned_7d": len(event_ids),
                    "symbols": symbols,
                    "event_links_inserted": inserted,
                }
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
