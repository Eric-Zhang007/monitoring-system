#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2.extras import execute_values
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_OPEN_INTEREST_URL = "https://fapi.binance.com/futures/data/openInterestHist"


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


def _to_dt_utc(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)


def _timeframe_seconds(timeframe: str) -> int:
    tf = str(timeframe or "5m").strip().lower()
    if tf.endswith("m"):
        return max(1, int(tf[:-1] or "1")) * 60
    if tf.endswith("h"):
        return max(1, int(tf[:-1] or "1")) * 3600
    if tf.endswith("d"):
        return max(1, int(tf[:-1] or "1")) * 86400
    raise ValueError(f"unsupported_timeframe:{timeframe}")


def _parse_symbols(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for p in str(raw or "").split(","):
        token = p.strip().upper().replace("$", "")
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _parse_symbol_map(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in str(raw or "").split(","):
        part = item.strip()
        if not part:
            continue
        if ":" in part:
            left, right = part.split(":", 1)
            sym = left.strip().upper()
            pair = right.strip().upper()
        else:
            sym = part.upper()
            pair = f"{sym}USDT"
        if sym:
            out[sym] = pair
    return out


def _build_session(proxy: str = "") -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    p = str(proxy or "").strip()
    if p:
        session.proxies.update({"http": p, "https": p})
    return session


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _fetch_funding_rates(
    session: requests.Session,
    *,
    pair: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    sleep_sec: float,
    max_pages: int,
) -> List[Tuple[int, float]]:
    rows: List[Tuple[int, float]] = []
    cursor = int(start_ms)
    pages = 0
    while cursor <= int(end_ms):
        params = {
            "symbol": str(pair),
            "startTime": int(cursor),
            "endTime": int(end_ms),
            "limit": int(max(1, min(1000, limit))),
        }
        resp = session.get(BINANCE_FUNDING_URL, params=params, timeout=25)
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, list):
            break
        if not body:
            break

        max_seen = cursor
        for item in body:
            if not isinstance(item, dict):
                continue
            ts_ms = int(item.get("fundingTime") or 0)
            if ts_ms <= 0:
                continue
            if ts_ms < int(start_ms) or ts_ms > int(end_ms):
                continue
            rows.append((ts_ms, _safe_float(item.get("fundingRate"), 0.0)))
            if ts_ms > max_seen:
                max_seen = ts_ms

        if max_seen <= cursor:
            break
        cursor = int(max_seen + 1)
        pages += 1
        if max_pages > 0 and pages >= int(max_pages):
            break
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))

    dedup: Dict[int, float] = {}
    for ts_ms, rate in rows:
        dedup[int(ts_ms)] = float(rate)
    return sorted(dedup.items(), key=lambda x: x[0])


def _build_funding_rows(
    raw_rows: Sequence[Tuple[int, float]],
    *,
    start_ms: int,
    end_ms: int,
    bucket_sec: int,
    expand_to_timeframe: bool,
) -> List[Tuple[int, float, Optional[int]]]:
    if not raw_rows:
        return []

    points = sorted((int(ts), float(rate)) for ts, rate in raw_rows if int(start_ms) <= int(ts) <= int(end_ms))
    if not points:
        return []

    if not bool(expand_to_timeframe):
        out_sparse: List[Tuple[int, float, Optional[int]]] = []
        for idx, (ts_ms, rate) in enumerate(points):
            nxt = points[idx + 1][0] if idx + 1 < len(points) else None
            out_sparse.append((int(ts_ms), float(rate), int(nxt) if nxt is not None else None))
        return out_sparse

    bucket_ms = int(max(1, bucket_sec)) * 1000
    out_dense: Dict[int, Tuple[float, Optional[int]]] = {}
    for idx, (ts_ms, rate) in enumerate(points):
        next_ts = points[idx + 1][0] if idx + 1 < len(points) else None
        cur = int(max(start_ms, ts_ms))
        cur = (cur // bucket_ms) * bucket_ms
        horizon = int(end_ms)
        if next_ts is not None:
            horizon = min(horizon, int(next_ts) - 1)
        while cur <= horizon:
            out_dense[int(cur)] = (float(rate), int(next_ts) if next_ts is not None else None)
            cur += bucket_ms

    rows: List[Tuple[int, float, Optional[int]]] = []
    for ts_ms in sorted(out_dense.keys()):
        rate, nxt = out_dense[ts_ms]
        rows.append((int(ts_ms), float(rate), int(nxt) if nxt is not None else None))
    return rows


def _fetch_open_interest(
    session: requests.Session,
    *,
    pair: str,
    period: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    sleep_sec: float,
    max_pages: int,
) -> Tuple[List[Tuple[int, float]], Optional[str]]:
    rows: List[Tuple[int, float]] = []
    cursor = int(start_ms)
    pages = 0
    warn: Optional[str] = None

    while cursor <= int(end_ms):
        params = {
            "symbol": str(pair),
            "period": str(period),
            "startTime": int(cursor),
            "endTime": int(end_ms),
            "limit": int(max(1, min(500, limit))),
        }
        resp = session.get(BINANCE_OPEN_INTEREST_URL, params=params, timeout=25)
        if resp.status_code >= 400:
            try:
                body_err = resp.json()
            except Exception:
                body_err = {"msg": (resp.text or "")[:200]}
            msg = str(body_err.get("msg") or body_err)
            if "startTime" in msg or "endTime" in msg:
                warn = f"open_interest_window_limited:{msg}"
                break
            resp.raise_for_status()

        body = resp.json()
        if not isinstance(body, list):
            break
        if not body:
            break

        max_seen = cursor
        for item in body:
            if not isinstance(item, dict):
                continue
            ts_ms = int(item.get("timestamp") or 0)
            if ts_ms <= 0:
                continue
            if ts_ms < int(start_ms) or ts_ms > int(end_ms):
                continue
            oi_val = _safe_float(item.get("sumOpenInterestValue"), math.nan)
            if not math.isfinite(oi_val):
                oi_val = _safe_float(item.get("sumOpenInterest"), math.nan)
            if not math.isfinite(oi_val):
                continue
            rows.append((ts_ms, float(oi_val)))
            if ts_ms > max_seen:
                max_seen = ts_ms

        if max_seen <= cursor:
            break
        cursor = int(max_seen + 1)
        pages += 1
        if max_pages > 0 and pages >= int(max_pages):
            break
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))

    dedup: Dict[int, float] = {}
    for ts_ms, value in rows:
        dedup[int(ts_ms)] = float(value)
    return sorted(dedup.items(), key=lambda x: x[0]), warn


def _build_open_interest_proxy_rows(
    raw_rows: Sequence[Tuple[int, float]],
    *,
    bucket_sec: int,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    if not raw_rows:
        return [], []
    bucket_ms = int(max(1, bucket_sec)) * 1000

    # Use last value in each bucket and convert level changes to net-inflow proxy.
    bucket_to_level: Dict[int, float] = {}
    for ts_ms, level in raw_rows:
        bucket = (int(ts_ms) // bucket_ms) * bucket_ms
        bucket_to_level[bucket] = float(level)

    buckets = sorted(bucket_to_level.keys())
    level_rows: List[Tuple[int, float]] = []
    inflow_rows: List[Tuple[int, float]] = []
    prev_level: Optional[float] = None
    for ts_ms in buckets:
        level = float(bucket_to_level[ts_ms])
        level_rows.append((int(ts_ms), level))
        if prev_level is None:
            inflow = 0.0
        else:
            inflow = float(level - prev_level)
        inflow_rows.append((int(ts_ms), inflow))
        prev_level = level
    return inflow_rows, level_rows


def _chunk_rows(rows: Sequence[Tuple[Any, ...]], size: int) -> Iterable[List[Tuple[Any, ...]]]:
    batch: List[Tuple[Any, ...]] = []
    for row in rows:
        batch.append(tuple(row))
        if len(batch) >= int(size):
            yield batch
            batch = []
    if batch:
        yield batch


def _upsert_funding_rows(
    conn,
    *,
    symbol: str,
    source: str,
    rows: Sequence[Tuple[int, float, Optional[int]]],
    batch_size: int,
) -> int:
    if not rows:
        return 0
    inserted = 0
    with conn.cursor() as cur:
        for batch in _chunk_rows(rows, size=max(200, int(batch_size))):
            values = [
                (
                    str(symbol).upper(),
                    _to_dt_utc(int(ts_ms)),
                    float(rate),
                    _to_dt_utc(int(next_ts)) if next_ts is not None else None,
                    str(source),
                )
                for ts_ms, rate, next_ts in batch
            ]
            execute_values(
                cur,
                """
                INSERT INTO funding_rates (
                    symbol, ts, funding_rate, next_funding_ts, source, created_at
                ) VALUES %s
                ON CONFLICT (symbol, ts)
                DO UPDATE SET
                    funding_rate = EXCLUDED.funding_rate,
                    next_funding_ts = EXCLUDED.next_funding_ts,
                    source = EXCLUDED.source
                """,
                values,
                template="(%s,%s,%s,%s,%s,NOW())",
                page_size=max(200, int(batch_size)),
            )
            inserted += len(values)
    return int(inserted)


def _insert_onchain_rows(
    conn,
    *,
    symbol: str,
    chain: str,
    metric_name: str,
    source: str,
    rows: Sequence[Tuple[int, float]],
    batch_size: int,
) -> int:
    if not rows:
        return 0
    inserted = 0
    with conn.cursor() as cur:
        for batch in _chunk_rows(rows, size=max(200, int(batch_size))):
            values = [
                (
                    str(symbol).upper(),
                    str(chain),
                    _to_dt_utc(int(ts_ms)),
                    str(metric_name),
                    float(metric_value),
                    str(source),
                )
                for ts_ms, metric_value in batch
            ]
            execute_values(
                cur,
                """
                INSERT INTO onchain_signals (
                    asset_symbol, chain, ts, metric_name, metric_value, source, created_at
                ) VALUES %s
                """,
                values,
                template="(%s,%s,%s,%s,%s,%s,NOW())",
                page_size=max(200, int(batch_size)),
            )
            inserted += len(values)
    return int(inserted)


def _fill_missing_onchain_from_market_bars(
    conn,
    *,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    metric_name: str,
    chain: str,
    source: str,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO onchain_signals (
                asset_symbol, chain, ts, metric_name, metric_value, source, created_at
            )
            SELECT
                mb.symbol AS asset_symbol,
                %s AS chain,
                mb.ts,
                %s AS metric_name,
                ((mb.close - mb.open) * GREATEST(0.0, mb.volume))::double precision AS metric_value,
                %s AS source,
                NOW()
            FROM market_bars mb
            LEFT JOIN onchain_signals os
              ON os.asset_symbol = mb.symbol
             AND os.ts = mb.ts
             AND os.metric_name = %s
            WHERE mb.symbol = %s
              AND mb.timeframe = %s
              AND mb.ts >= %s
              AND mb.ts <= %s
              AND os.id IS NULL
            """,
            (
                str(chain),
                str(metric_name),
                str(source),
                str(metric_name),
                str(symbol).upper(),
                str(timeframe),
                start,
                end,
            ),
        )
        return int(cur.rowcount or 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest Binance funding + open-interest proxy into funding_rates/onchain_signals")
    ap.add_argument("--database-url", default=DATABASE_URL)
    ap.add_argument("--symbols", default="BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK")
    ap.add_argument(
        "--symbol-map",
        default="BTC:BTCUSDT,ETH:ETHUSDT,SOL:SOLUSDT,BNB:BNBUSDT,XRP:XRPUSDT,ADA:ADAUSDT,DOGE:DOGEUSDT,TRX:TRXUSDT,AVAX:AVAXUSDT,LINK:LINKUSDT",
    )
    ap.add_argument("--start", default="2018-01-01T00:00:00Z")
    ap.add_argument("--end", default="")
    ap.add_argument("--timeframe", default=os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m"))
    ap.add_argument("--oi-period", default="5m", help="Binance openInterestHist period")
    ap.add_argument("--oi-max-lookback-days", type=int, default=int(os.getenv("BINANCE_OI_MAX_LOOKBACK_DAYS", "30")))
    ap.add_argument("--funding-limit", type=int, default=1000)
    ap.add_argument("--oi-limit", type=int, default=500)
    ap.add_argument("--max-funding-pages", type=int, default=0)
    ap.add_argument("--max-oi-pages", type=int, default=0)
    ap.add_argument("--sleep-sec", type=float, default=0.08)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--source-funding", default="binance_fapi_funding")
    ap.add_argument("--source-oi", default="binance_fapi_open_interest")
    ap.add_argument("--source-oi-proxy", default="binance_fapi_oi_proxy")
    ap.add_argument("--source-oi-fallback", default="synthetic_oi_proxy_from_bars")
    ap.add_argument("--metric-name", default="net_inflow")
    ap.add_argument("--chain", default="derivatives")
    ap.add_argument("--expand-funding-to-timeframe", action="store_true", default=True)
    ap.add_argument("--no-expand-funding-to-timeframe", dest="expand_funding_to_timeframe", action="store_false")
    ap.add_argument("--store-oi-level", action="store_true", default=True)
    ap.add_argument("--no-store-oi-level", dest="store_oi_level", action="store_false")
    ap.add_argument("--fill-missing-onchain-from-market-bars", action="store_true", default=True)
    ap.add_argument("--no-fill-missing-onchain-from-market-bars", dest="fill_missing_onchain_from_market_bars", action="store_false")
    ap.add_argument("--replace-window", action="store_true", default=True)
    ap.add_argument("--no-replace-window", dest="replace_window", action="store_false")
    ap.add_argument("--proxy", default=os.getenv("HTTPS_PROXY", os.getenv("ALL_PROXY", "")))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise RuntimeError("empty_symbols")
    symbol_map = _parse_symbol_map(args.symbol_map)

    start_dt = _parse_dt_utc(args.start)
    end_dt = _parse_dt_utc(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    if end_dt <= start_dt:
        raise RuntimeError("invalid_time_range")

    timeframe_sec = _timeframe_seconds(args.timeframe)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    oi_max_lookback_days = max(1, int(args.oi_max_lookback_days))
    oi_floor_dt = datetime.now(timezone.utc) - timedelta(days=oi_max_lookback_days)
    oi_floor_ms = int(oi_floor_dt.timestamp() * 1000)

    session = _build_session(proxy=str(args.proxy))

    out: Dict[str, Any] = {
        "status": "ok",
        "start": _to_iso_z(start_dt),
        "end": _to_iso_z(end_dt),
        "timeframe": str(args.timeframe),
        "oi_period": str(args.oi_period),
        "symbols": symbols,
        "replace_window": bool(args.replace_window),
        "expand_funding_to_timeframe": bool(args.expand_funding_to_timeframe),
        "fill_missing_onchain_from_market_bars": bool(args.fill_missing_onchain_from_market_bars),
        "dry_run": bool(args.dry_run),
        "per_symbol": {},
        "warnings": [],
    }

    if bool(args.dry_run):
        for sym in symbols:
            pair = symbol_map.get(sym, f"{sym}USDT")
            oi_clamped = max(start_ms, oi_floor_ms)
            out["per_symbol"][sym] = {
                "pair": pair,
                "funding_fetch_window": {"start": _to_iso_z(_to_dt_utc(start_ms)), "end": _to_iso_z(_to_dt_utc(end_ms))},
                "oi_fetch_window": {
                    "start": _to_iso_z(_to_dt_utc(oi_clamped)),
                    "end": _to_iso_z(_to_dt_utc(end_ms)),
                },
            }
            if oi_clamped > start_ms:
                out["warnings"].append(
                    f"{sym}: open-interest window clamped to >= {_to_iso_z(_to_dt_utc(oi_clamped))} due to exchange retention limit"
                )
        out_text = json.dumps(out, ensure_ascii=False)
        if str(args.out_json).strip():
            os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
            with open(str(args.out_json), "w", encoding="utf-8") as f:
                f.write(out_text + "\n")
        print(out_text)
        return 0

    with psycopg2.connect(args.database_url) as conn:
        conn.autocommit = False
        for sym in symbols:
            pair = symbol_map.get(sym, f"{sym}USDT")
            sym_summary: Dict[str, Any] = {"pair": pair, "warnings": []}

            if bool(args.replace_window):
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        DELETE FROM funding_rates
                        WHERE symbol = %s
                          AND ts >= %s
                          AND ts <= %s
                          AND source = %s
                        """,
                        (sym, start_dt, end_dt, str(args.source_funding)),
                    )
                    sym_summary["funding_deleted_rows"] = int(cur.rowcount or 0)
                    cur.execute(
                        """
                        DELETE FROM onchain_signals
                        WHERE asset_symbol = %s
                          AND ts >= %s
                          AND ts <= %s
                          AND chain = %s
                          AND metric_name IN (%s, 'open_interest_value')
                          AND source IN (%s, %s)
                        """,
                        (
                            sym,
                            start_dt,
                            end_dt,
                            str(args.chain),
                            str(args.metric_name),
                            str(args.source_oi),
                            str(args.source_oi_proxy),
                        ),
                    )
                    sym_summary["onchain_deleted_rows"] = int(cur.rowcount or 0)
                    if bool(args.fill_missing_onchain_from_market_bars):
                        cur.execute(
                            """
                            DELETE FROM onchain_signals
                            WHERE asset_symbol = %s
                              AND ts >= %s
                              AND ts <= %s
                              AND chain = %s
                              AND metric_name = %s
                              AND source = %s
                            """,
                            (sym, start_dt, end_dt, str(args.chain), str(args.metric_name), str(args.source_oi_fallback)),
                        )
                        sym_summary["fallback_deleted_rows"] = int(cur.rowcount or 0)

            funding_raw = _fetch_funding_rates(
                session,
                pair=pair,
                start_ms=start_ms,
                end_ms=end_ms,
                limit=int(args.funding_limit),
                sleep_sec=float(args.sleep_sec),
                max_pages=int(args.max_funding_pages),
            )
            funding_rows = _build_funding_rows(
                funding_raw,
                start_ms=start_ms,
                end_ms=end_ms,
                bucket_sec=timeframe_sec,
                expand_to_timeframe=bool(args.expand_funding_to_timeframe),
            )
            funding_written = _upsert_funding_rows(
                conn,
                symbol=sym,
                source=str(args.source_funding),
                rows=funding_rows,
                batch_size=int(args.batch_size),
            )

            oi_start_ms = max(start_ms, oi_floor_ms)
            if oi_start_ms > start_ms:
                msg = f"open-interest window clamped to >= {_to_iso_z(_to_dt_utc(oi_start_ms))}"
                sym_summary["warnings"].append(msg)
                out["warnings"].append(f"{sym}: {msg}")

            oi_raw: List[Tuple[int, float]] = []
            oi_warn: Optional[str] = None
            if oi_start_ms <= end_ms:
                oi_raw, oi_warn = _fetch_open_interest(
                    session,
                    pair=pair,
                    period=str(args.oi_period),
                    start_ms=oi_start_ms,
                    end_ms=end_ms,
                    limit=int(args.oi_limit),
                    sleep_sec=float(args.sleep_sec),
                    max_pages=int(args.max_oi_pages),
                )
                if oi_warn:
                    sym_summary["warnings"].append(str(oi_warn))
                    out["warnings"].append(f"{sym}: {oi_warn}")

            oi_inflow_rows, oi_level_rows = _build_open_interest_proxy_rows(oi_raw, bucket_sec=timeframe_sec)
            oi_proxy_written = _insert_onchain_rows(
                conn,
                symbol=sym,
                chain=str(args.chain),
                metric_name=str(args.metric_name),
                source=str(args.source_oi_proxy),
                rows=oi_inflow_rows,
                batch_size=int(args.batch_size),
            )
            oi_level_written = 0
            if bool(args.store_oi_level):
                oi_level_written = _insert_onchain_rows(
                    conn,
                    symbol=sym,
                    chain=str(args.chain),
                    metric_name="open_interest_value",
                    source=str(args.source_oi),
                    rows=oi_level_rows,
                    batch_size=int(args.batch_size),
                )

            fallback_written = 0
            if bool(args.fill_missing_onchain_from_market_bars):
                fallback_written = _fill_missing_onchain_from_market_bars(
                    conn,
                    symbol=sym,
                    timeframe=str(args.timeframe),
                    start=start_dt,
                    end=end_dt,
                    metric_name=str(args.metric_name),
                    chain=str(args.chain),
                    source=str(args.source_oi_fallback),
                )

            sym_summary.update(
                {
                    "funding_raw_points": len(funding_raw),
                    "funding_rows_written": int(funding_written),
                    "oi_start": _to_iso_z(_to_dt_utc(oi_start_ms)),
                    "oi_end": _to_iso_z(_to_dt_utc(end_ms)),
                    "oi_raw_points": len(oi_raw),
                    "oi_proxy_rows_written": int(oi_proxy_written),
                    "oi_level_rows_written": int(oi_level_written),
                    "oi_fallback_rows_written": int(fallback_written),
                }
            )
            out["per_symbol"][sym] = sym_summary
            conn.commit()

    out_text = json.dumps(out, ensure_ascii=False)
    if str(args.out_json).strip():
        os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
        with open(str(args.out_json), "w", encoding="utf-8") as f:
            f.write(out_text + "\n")
    print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
