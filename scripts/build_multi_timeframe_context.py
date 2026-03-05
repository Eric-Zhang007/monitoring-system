#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from bisect import bisect_right
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if not text:
        return datetime.now(timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _norm_symbols(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for part in str(raw or "").split(","):
        sym = str(part or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _norm_timeframes(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for part in str(raw or "").split(","):
        tf = str(part or "").strip().lower()
        if not tf or tf in seen:
            continue
        ok = (tf.endswith("m") and tf[:-1].isdigit()) or (tf.endswith("h") and tf[:-1].isdigit()) or (tf.endswith("d") and tf[:-1].isdigit())
        if not ok:
            continue
        seen.add(tf)
        out.append(tf)
    return out


def _source_hash(primary_tf: str, context_tfs: List[str], start: datetime, end: datetime) -> str:
    body = json.dumps(
        {
            "primary_timeframe": primary_tf,
            "context_timeframes": context_tfs,
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _asof_row(ts_list: List[datetime], rows: List[Dict[str, Any]], as_of: datetime) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    if not ts_list:
        return None, None
    idx = bisect_right(ts_list, as_of) - 1
    if idx < 0:
        return None, None
    cur = rows[idx]
    prev = rows[idx - 1] if idx - 1 >= 0 else None
    return cur, prev


def main() -> int:
    ap = argparse.ArgumentParser(description="Build aligned multi-timeframe market context table from market_bars")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    ap.add_argument("--start", default="2018-01-01T00:00:00Z")
    ap.add_argument("--end", default="")
    ap.add_argument("--primary-timeframe", default=os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m"))
    ap.add_argument("--context-timeframes", default=os.getenv("ANALYST_CONTEXT_TIMEFRAMES", "5m,15m,1h,4h,1d"))
    ap.add_argument("--batch-size", type=int, default=2000)
    args = ap.parse_args()

    start_dt = _parse_dt(args.start)
    end_dt = _parse_dt(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    symbols = _norm_symbols(args.symbols)
    primary_tf = str(args.primary_timeframe).strip().lower()
    context_tfs = _norm_timeframes(args.context_timeframes)
    if primary_tf not in context_tfs:
        context_tfs.insert(0, primary_tf)
    if not symbols:
        raise RuntimeError("empty_symbols")
    if not context_tfs:
        raise RuntimeError("empty_context_timeframes")

    src_hash = _source_hash(primary_tf, context_tfs, start_dt, end_dt)
    total_rows = 0
    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS market_context_multi_tf (
                    symbol VARCHAR(32) NOT NULL,
                    as_of_ts TIMESTAMPTZ NOT NULL,
                    primary_timeframe VARCHAR(8) NOT NULL,
                    context_json JSONB NOT NULL,
                    coverage_json JSONB NOT NULL,
                    source_hash VARCHAR(64) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY(symbol, primary_timeframe, as_of_ts)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_market_context_multi_tf_symbol_ts ON market_context_multi_tf(symbol, as_of_ts DESC)")

            for sym in symbols:
                tf_rows: Dict[str, List[Dict[str, Any]]] = {}
                tf_ts: Dict[str, List[datetime]] = {}
                for tf in context_tfs:
                    cur.execute(
                        """
                        SELECT ts, close::float AS close, volume::float AS volume
                        FROM market_bars
                        WHERE symbol = %s
                          AND timeframe = %s
                          AND ts >= %s
                          AND ts <= %s
                        ORDER BY ts ASC
                        """,
                        (sym, tf, start_dt, end_dt),
                    )
                    rows = [dict(r) for r in cur.fetchall()]
                    tf_rows[tf] = rows
                    tf_ts[tf] = [r["ts"] for r in rows if isinstance(r.get("ts"), datetime)]

                primary_rows = tf_rows.get(primary_tf) or []
                if not primary_rows:
                    continue

                to_upsert: List[Tuple[Any, ...]] = []
                for pr in primary_rows:
                    as_of = pr.get("ts")
                    if not isinstance(as_of, datetime):
                        continue
                    ctx_obj: Dict[str, Dict[str, Any]] = {}
                    cov_obj: Dict[str, Dict[str, Any]] = {}
                    for tf in context_tfs:
                        cur_row, prev_row = _asof_row(tf_ts.get(tf, []), tf_rows.get(tf, []), as_of)
                        if not cur_row:
                            ctx_obj[tf] = {"missing": 1, "close": 0.0, "ret_1": 0.0, "volume": 0.0, "as_of_ts": None, "lag_sec": 0.0}
                            cov_obj[tf] = {"missing": 1, "asof_lag_sec": None}
                            continue
                        close0 = float(cur_row.get("close") or 0.0)
                        vol0 = float(cur_row.get("volume") or 0.0)
                        ret_1 = 0.0
                        if prev_row is not None:
                            prev_close = float(prev_row.get("close") or 0.0)
                            if abs(prev_close) > 1e-12:
                                ret_1 = float((close0 / prev_close) - 1.0)
                        lag_sec = max(0.0, (as_of - cur_row["ts"]).total_seconds())
                        ctx_obj[tf] = {
                            "missing": 0,
                            "close": close0,
                            "ret_1": ret_1,
                            "volume": vol0,
                            "as_of_ts": cur_row["ts"].isoformat(),
                            "lag_sec": float(lag_sec),
                        }
                        cov_obj[tf] = {"missing": 0, "asof_lag_sec": float(lag_sec)}
                    to_upsert.append((sym, as_of, primary_tf, json.dumps(ctx_obj), json.dumps(cov_obj), src_hash))

                if not to_upsert:
                    continue
                execute_values(
                    cur,
                    """
                    INSERT INTO market_context_multi_tf (
                        symbol, as_of_ts, primary_timeframe, context_json, coverage_json, source_hash
                    ) VALUES %s
                    ON CONFLICT (symbol, primary_timeframe, as_of_ts)
                    DO UPDATE SET
                        context_json = EXCLUDED.context_json,
                        coverage_json = EXCLUDED.coverage_json,
                        source_hash = EXCLUDED.source_hash,
                        updated_at = NOW()
                    """,
                    to_upsert,
                    page_size=max(200, int(args.batch_size)),
                )
                total_rows += int(len(to_upsert))
        conn.commit()

    print(
        json.dumps(
            {
                "status": "ok",
                "symbols": symbols,
                "primary_timeframe": primary_tf,
                "context_timeframes": context_tfs,
                "rows_upserted": int(total_rows),
                "source_hash": src_hash,
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
