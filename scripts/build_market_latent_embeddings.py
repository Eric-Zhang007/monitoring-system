#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _pad_tail(values: np.ndarray, size: int) -> np.ndarray:
    if values.size >= size:
        return values[-size:]
    if values.size <= 0:
        return np.zeros((size,), dtype=np.float64)
    out = np.zeros((size,), dtype=np.float64)
    out[-values.size :] = values
    return out


def _safe_stats(values: np.ndarray) -> List[float]:
    if values.size <= 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mean = float(np.mean(values))
    std = float(np.std(values))
    mn = float(np.min(values))
    mx = float(np.max(values))
    q10 = float(np.quantile(values, 0.1))
    q90 = float(np.quantile(values, 0.9))
    return [mean, std, mn, mx, q10, q90]


def _latent_vec(prices: List[float], vols: List[float], idx: int, dim: int = 128) -> List[float]:
    p = np.array(prices[max(0, idx - 512) : idx + 1], dtype=np.float64)
    if p.size < 32:
        return [0.0] * dim
    r = np.diff(np.log(np.clip(p, 1e-12, None)))
    abs_r = np.abs(r)
    v = np.array(vols[max(0, idx - 512) : idx + 1], dtype=np.float64)
    lv = np.diff(np.log1p(np.clip(v, 0.0, None)))

    out: List[float] = []
    out.extend(np.tanh(_pad_tail(r, 32) * 8.0).tolist())
    out.extend(np.tanh(_pad_tail(abs_r, 32) * 6.0).tolist())

    for w in (3, 6, 12, 24, 48, 96, 192):
        seg = r[-min(w, r.size) :] if r.size > 0 else np.array([], dtype=np.float64)
        out.extend(_safe_stats(seg))
    for w in (3, 6, 12, 24, 48, 96):
        seg = lv[-min(w, lv.size) :] if lv.size > 0 else np.array([], dtype=np.float64)
        out.extend(_safe_stats(seg))

    # Cross interactions between return and volume dynamics.
    cross = 0.0
    if r.size > 2 and lv.size > 2:
        n = min(r.size, lv.size)
        if n > 2:
            rr = r[-n:]
            vv = lv[-n:]
            cross = float(np.corrcoef(rr, vv)[0, 1]) if float(np.std(rr)) > 1e-12 and float(np.std(vv)) > 1e-12 else 0.0
    out.append(float(np.clip(cross, -1.0, 1.0)))
    out.append(float(np.tanh(float(np.mean(abs_r[-24:])) * 10.0 if abs_r.size else 0.0)))

    if len(out) < dim:
        out.extend([0.0] * (dim - len(out)))
    return [float(x) for x in out[:dim]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build market latent embeddings")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--start", default="2018-01-01T00:00:00Z")
    parser.add_argument("--end", default="")
    parser.add_argument("--timeframe", default=os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m"))
    parser.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--max-rows-per-symbol", type=int, default=int(os.getenv("MARKET_LATENT_MAX_ROWS_PER_SYMBOL", "0")))
    args = parser.parse_args()

    start_dt = _parse_dt(args.start)
    end_dt = _parse_dt(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]

    inserted = 0

    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS market_latent (
                    id BIGSERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    as_of_ts TIMESTAMPTZ NOT NULL,
                    latent JSONB NOT NULL,
                    latent_dim INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_market_latent_symbol_ts ON market_latent(symbol, as_of_ts DESC)")
            if bool(args.truncate):
                cur.execute("TRUNCATE TABLE market_latent")

            max_rows = int(args.max_rows_per_symbol)
            for sym in symbols:
                sql = """
                    SELECT ts, close::double precision AS close, volume::double precision AS volume
                    FROM market_bars
                    WHERE symbol = %s
                      AND timeframe = %s
                      AND ts >= %s
                      AND ts <= %s
                    ORDER BY ts ASC
                """
                params: List[object] = [sym, str(args.timeframe), start_dt, end_dt]
                if max_rows > 0:
                    sql += " LIMIT %s"
                    params.append(max_rows)
                cur.execute(sql, tuple(params))
                rows = [dict(r) for r in cur.fetchall()]
                if len(rows) < 24:
                    continue

                prices = [float(r.get("close") or 0.0) for r in rows]
                vols = [float(r.get("volume") or 0.0) for r in rows]
                ts_list = [r.get("ts") for r in rows]
                data_rows = []
                for i in range(12, len(rows)):
                    p = float(prices[i] or 0.0)
                    if p <= 0:
                        continue
                    latent = _latent_vec(prices, vols, i, dim=128)
                    data_rows.append((sym, ts_list[i], json.dumps(latent), len(latent)))

                if data_rows:
                    execute_values(
                        cur,
                        """
                        INSERT INTO market_latent(symbol, as_of_ts, latent, latent_dim)
                        VALUES %s
                        """,
                        data_rows,
                        template="(%s, %s, %s::jsonb, %s)",
                    )
                    inserted += len(data_rows)

        conn.commit()

    print(
        json.dumps(
            {
                "status": "ok",
                "table": "market_latent",
                "rows_inserted": int(inserted),
                "latent_dim": 128,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
