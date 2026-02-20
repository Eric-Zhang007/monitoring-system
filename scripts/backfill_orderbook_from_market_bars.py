#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")


def _parse_dt_utc(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty_datetime")
    norm = text.replace(" ", "T")
    if norm.endswith("Z"):
        norm = norm[:-1] + "+00:00"
    dt = datetime.fromisoformat(norm)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_symbols(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in str(raw or "").split(","):
        sym = item.strip().upper().replace("$", "")
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def _table_exists(cur, table: str) -> bool:
    cur.execute("SELECT to_regclass(%s) AS reg", (f"public.{table}",))
    row = cur.fetchone() or {}
    return bool(row.get("reg"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill synthetic orderbook_l2 rows from market_bars")
    ap.add_argument("--database-url", default=DATABASE_URL)
    ap.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK"))
    ap.add_argument("--timeframe", default=os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m"))
    ap.add_argument("--start", default="2018-01-01T00:00:00Z")
    ap.add_argument("--end", default="")
    ap.add_argument("--source", default="synthetic_from_market_bars")
    ap.add_argument("--replace-window", action="store_true", default=True)
    ap.add_argument("--no-replace-window", dest="replace_window", action="store_false")
    ap.add_argument("--min-spread-bps", type=float, default=float(os.getenv("OB_PROXY_MIN_SPREAD_BPS", "1.5")))
    ap.add_argument("--max-spread-bps", type=float, default=float(os.getenv("OB_PROXY_MAX_SPREAD_BPS", "35.0")))
    ap.add_argument("--spread-ret-k", type=float, default=float(os.getenv("OB_PROXY_SPREAD_RET_K", "2800.0")))
    ap.add_argument("--spread-depth-k", type=float, default=float(os.getenv("OB_PROXY_SPREAD_DEPTH_K", "8.0")))
    ap.add_argument("--base-size-div", type=float, default=float(os.getenv("OB_PROXY_BASE_SIZE_DIV", "20.0")))
    ap.add_argument("--min-base-size", type=float, default=float(os.getenv("OB_PROXY_MIN_BASE_SIZE", "1.0")))
    ap.add_argument("--imbalance-k", type=float, default=float(os.getenv("OB_PROXY_IMBALANCE_K", "2.0")))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise RuntimeError("empty_symbols")

    start_dt = _parse_dt_utc(args.start)
    end_dt = _parse_dt_utc(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    if end_dt <= start_dt:
        raise RuntimeError("invalid_time_range")

    if float(args.max_spread_bps) <= float(args.min_spread_bps):
        raise RuntimeError("max_spread_bps_must_be_gt_min_spread_bps")

    out: Dict[str, Any] = {
        "status": "ok",
        "source": str(args.source),
        "symbols": symbols,
        "timeframe": str(args.timeframe),
        "window": {"start": _to_iso_z(start_dt), "end": _to_iso_z(end_dt)},
        "replace_window": bool(args.replace_window),
        "dry_run": bool(args.dry_run),
    }

    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            if not _table_exists(cur, "market_bars"):
                raise RuntimeError("missing_table:market_bars")
            if not _table_exists(cur, "orderbook_l2"):
                raise RuntimeError("missing_table:orderbook_l2")

            cur.execute(
                """
                SELECT COUNT(*)::bigint AS n
                FROM market_bars
                WHERE timeframe = %s
                  AND symbol = ANY(%s)
                  AND ts >= %s
                  AND ts <= %s
                """,
                (str(args.timeframe), symbols, start_dt, end_dt),
            )
            src_rows = int((cur.fetchone() or {}).get("n") or 0)
            out["market_bars_rows"] = int(src_rows)

            cur.execute(
                """
                SELECT COUNT(*)::bigint AS n
                FROM orderbook_l2
                WHERE symbol = ANY(%s)
                  AND ts >= %s
                  AND ts <= %s
                """,
                (symbols, start_dt, end_dt),
            )
            out["orderbook_rows_before"] = int((cur.fetchone() or {}).get("n") or 0)

            if bool(args.dry_run):
                out["orderbook_rows_deleted"] = 0
                out["orderbook_rows_inserted"] = 0
                out["orderbook_rows_after"] = out["orderbook_rows_before"]
                out_text = json.dumps(out, ensure_ascii=False)
                if str(args.out_json).strip():
                    os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
                    with open(str(args.out_json), "w", encoding="utf-8") as f:
                        f.write(out_text + "\n")
                print(out_text)
                return 0

            deleted = 0
            if bool(args.replace_window):
                cur.execute(
                    """
                    DELETE FROM orderbook_l2
                    WHERE symbol = ANY(%s)
                      AND ts >= %s
                      AND ts <= %s
                      AND source = %s
                    """,
                    (symbols, start_dt, end_dt, str(args.source)),
                )
                deleted = int(cur.rowcount or 0)

            cur.execute(
                """
                INSERT INTO orderbook_l2 (
                    symbol, ts, bid_px, ask_px, bid_sz, ask_sz, spread_bps, imbalance, source, created_at
                )
                WITH src AS (
                    SELECT
                        mb.symbol,
                        mb.ts,
                        mb.open,
                        mb.high,
                        mb.low,
                        mb.close,
                        COALESCE(mb.volume, 0.0) AS volume
                    FROM market_bars mb
                    WHERE mb.timeframe = %s
                      AND mb.symbol = ANY(%s)
                      AND mb.ts >= %s
                      AND mb.ts <= %s
                ),
                f AS (
                    SELECT
                        s.symbol,
                        s.ts,
                        s.open,
                        s.high,
                        s.low,
                        s.close,
                        s.volume,
                        LEAST(%s::double precision,
                              GREATEST(%s::double precision,
                                       %s::double precision
                                       + ABS(LN(GREATEST(s.close, 1e-9) / GREATEST(s.open, 1e-9))) * %s::double precision
                                       + (%s::double precision / GREATEST(1.0, LN(2.0 + GREATEST(s.volume, 0.0))))
                              )
                        ) AS spread_bps,
                        GREATEST(%s::double precision, GREATEST(0.0, s.volume) / %s::double precision) AS base_size,
                        GREATEST(-0.95, LEAST(0.95,
                            TANH(
                                %s::double precision
                                * ((s.close - s.open) / GREATEST(ABS(s.high - s.low), GREATEST(s.close, 1e-6) * 0.0005))
                            )
                        )) AS imbalance
                    FROM src s
                )
                SELECT
                    f.symbol,
                    f.ts,
                    (f.close * (1.0 - f.spread_bps / 20000.0))::double precision AS bid_px,
                    (f.close * (1.0 + f.spread_bps / 20000.0))::double precision AS ask_px,
                    (f.base_size * (1.0 + f.imbalance))::double precision AS bid_sz,
                    (f.base_size * (1.0 - f.imbalance))::double precision AS ask_sz,
                    f.spread_bps,
                    f.imbalance,
                    %s,
                    NOW()
                FROM f
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM orderbook_l2 ob
                    WHERE ob.symbol = f.symbol
                      AND ob.ts = f.ts
                )
                """,
                (
                    str(args.timeframe),
                    symbols,
                    start_dt,
                    end_dt,
                    float(args.max_spread_bps),
                    float(args.min_spread_bps),
                    float(args.min_spread_bps),
                    float(args.spread_ret_k),
                    float(args.spread_depth_k),
                    float(args.min_base_size),
                    float(args.base_size_div),
                    float(args.imbalance_k),
                    str(args.source),
                ),
            )
            inserted = int(cur.rowcount or 0)

            cur.execute(
                """
                SELECT COUNT(*)::bigint AS n
                FROM orderbook_l2
                WHERE symbol = ANY(%s)
                  AND ts >= %s
                  AND ts <= %s
                """,
                (symbols, start_dt, end_dt),
            )
            after_rows = int((cur.fetchone() or {}).get("n") or 0)

        conn.commit()

    out["orderbook_rows_deleted"] = int(deleted)
    out["orderbook_rows_inserted"] = int(inserted)
    out["orderbook_rows_after"] = int(after_rows)
    out_text = json.dumps(out, ensure_ascii=False)
    if str(args.out_json).strip():
        os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
        with open(str(args.out_json), "w", encoding="utf-8") as f:
            f.write(out_text + "\n")
    print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
