#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from inference.liquid_feature_contract import LIQUID_MANUAL_FEATURE_KEYS, LIQUID_FEATURE_SCHEMA_VERSION


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _build_manual_payload(
    *,
    price: float,
    prev_1: float,
    prev_3: float,
    prev_12: float,
    prev_48: float,
    volume: float,
    row_index: int,
) -> Dict[str, float]:
    ret_1 = (price - prev_1) / max(prev_1, 1e-12)
    ret_3 = (price - prev_3) / max(prev_3, 1e-12)
    ret_12 = (price - prev_12) / max(prev_12, 1e-12)
    ret_48 = (price - prev_48) / max(prev_48, 1e-12)
    vol_z = math.tanh(math.log1p(max(0.0, volume)) / 8.0)

    out = {k: 0.0 for k in LIQUID_MANUAL_FEATURE_KEYS}
    base = {
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_12": ret_12,
        "ret_48": ret_48,
        "vol_3": abs(ret_1),
        "vol_12": abs(ret_3),
        "vol_48": abs(ret_12),
        "vol_96": abs(ret_48),
        "log_volume": math.log1p(max(0.0, volume)),
        "vol_z": vol_z,
        "volume_impact": abs(ret_1) / max(math.sqrt(max(1.0, volume)), 1e-6),
        "orderbook_imbalance": 0.0,
        "funding_rate": 0.0,
        "onchain_norm": 0.0,
        "event_decay": 0.0,
        "orderbook_missing_flag": 1.0,
        "funding_missing_flag": 1.0,
        "onchain_missing_flag": 1.0,
        "source_tier_weight": 0.0,
        "source_confidence": 0.0,
        "social_post_sentiment": 0.0,
        "social_comment_sentiment": 0.0,
        "social_engagement_norm": 0.0,
        "social_influence_norm": 0.0,
        "social_event_ratio": 0.0,
        "social_buzz": 0.0,
        "event_velocity_1h": 0.0,
        "event_velocity_6h": 0.0,
        "event_disagreement": 0.0,
        "source_diversity": 0.0,
        "cross_source_consensus": 0.0,
        "comment_skew": 0.0,
        "event_lag_bucket_0_1h": 0.0,
        "event_lag_bucket_1_6h": 0.0,
        "event_lag_bucket_6_24h": 0.0,
    }
    for k, v in base.items():
        if k in out:
            out[k] = float(v)

    stat_keys = [k for k in LIQUID_MANUAL_FEATURE_KEYS if k.startswith("manual_stat_")]
    for idx, k in enumerate(stat_keys):
        alpha = (idx % 11) + 1
        beta = (idx % 17) + 1
        gamma = (idx % 5) + 1
        signal = (
            alpha * ret_1
            + 0.7 * beta * ret_3
            + 0.3 * gamma * ret_12
            + 0.2 * ret_48
            + 0.05 * vol_z
            + 0.01 * math.sin((row_index + 1) * (idx + 1) * 0.003)
        )
        out[k] = float(math.tanh(signal))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build feature snapshots from full-history market bars")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--start", default="2018-01-01T00:00:00Z")
    parser.add_argument("--end", default="")
    parser.add_argument("--timeframe", default=os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m"))
    parser.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    parser.add_argument("--max-rows-per-symbol", type=int, default=int(os.getenv("FEATURE_MAIN_MAX_ROWS_PER_SYMBOL", "0")))
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()

    start_dt = _parse_dt(args.start)
    end_dt = _parse_dt(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]

    created = 0

    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_snapshots_main (
                    id BIGSERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    as_of_ts TIMESTAMPTZ NOT NULL,
                    feature_payload JSONB NOT NULL,
                    feature_dim INTEGER NOT NULL,
                    feature_version TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feature_snapshots_main_symbol_ts ON feature_snapshots_main(symbol, as_of_ts DESC)")
            if bool(args.truncate):
                cur.execute("TRUNCATE TABLE feature_snapshots_main")

            max_rows = int(args.max_rows_per_symbol)
            for sym in symbols:
                # Idempotency: rebuilding same symbol/window/version should not accumulate duplicates.
                cur.execute(
                    """
                    DELETE FROM feature_snapshots_main
                    WHERE symbol = %s
                      AND as_of_ts >= %s
                      AND as_of_ts <= %s
                      AND feature_version = %s
                    """,
                    (sym, start_dt, end_dt, str(LIQUID_FEATURE_SCHEMA_VERSION)),
                )
                sql = """
                    SELECT ts, close::double precision AS close, volume::double precision AS volume
                    FROM market_bars
                    WHERE symbol = %s
                      AND timeframe = %s
                      AND ts >= %s
                      AND ts <= %s
                    ORDER BY ts ASC
                """
                params: List[Any] = [sym, str(args.timeframe), start_dt, end_dt]
                if max_rows > 0:
                    sql += " LIMIT %s"
                    params.append(max_rows)
                cur.execute(sql, tuple(params))
                rows = [dict(r) for r in cur.fetchall()]
                if len(rows) < 64:
                    continue

                payload_rows: List[tuple] = []
                prices = [float(r.get("close") or 0.0) for r in rows]
                vols = [float(r.get("volume") or 0.0) for r in rows]
                ts_list = [r.get("ts") for r in rows]
                for i in range(48, len(rows)):
                    p = float(prices[i])
                    if p <= 0:
                        continue
                    payload = _build_manual_payload(
                        price=p,
                        prev_1=float(prices[i - 1] or p),
                        prev_3=float(prices[i - 3] or p),
                        prev_12=float(prices[i - 12] or p),
                        prev_48=float(prices[i - 48] or p),
                        volume=float(vols[i] or 0.0),
                        row_index=i,
                    )
                    payload_rows.append(
                        (
                            sym,
                            ts_list[i],
                            json.dumps(payload),
                            len(LIQUID_MANUAL_FEATURE_KEYS),
                            str(LIQUID_FEATURE_SCHEMA_VERSION),
                        )
                    )

                if payload_rows:
                    execute_values(
                        cur,
                        """
                        INSERT INTO feature_snapshots_main(symbol, as_of_ts, feature_payload, feature_dim, feature_version)
                        VALUES %s
                        """,
                        payload_rows,
                        template="(%s, %s, %s::jsonb, %s, %s)",
                    )
                    created += len(payload_rows)

        conn.commit()

    print(
        json.dumps(
            {
                "status": "ok",
                "table": "feature_snapshots_main",
                "rows_created": int(created),
                "manual_feature_dim": len(LIQUID_MANUAL_FEATURE_KEYS),
                "feature_version": str(LIQUID_FEATURE_SCHEMA_VERSION),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
