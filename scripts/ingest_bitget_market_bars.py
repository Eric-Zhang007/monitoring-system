#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import time
from typing import Any, Dict, Iterable, List, Tuple

import psycopg2
from psycopg2.extras import execute_values
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BITGET_PERP_URL = "https://api.bitget.com/api/v2/mix/market/history-candles"
BITGET_SPOT_URL = "https://api.bitget.com/api/v2/spot/market/history-candles"
COINGECKO_MARKET_CHART_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")


@dataclass
class Bar:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _timeframe_minutes(tf: str) -> int:
    t = str(tf or "1h").strip().lower()
    if t.endswith("m"):
        return max(1, int(t[:-1] or "1"))
    if t.endswith("h"):
        return max(1, int(t[:-1] or "1")) * 60
    if t.endswith("d"):
        return max(1, int(t[:-1] or "1")) * 1440
    raise ValueError(f"unsupported_timeframe:{tf}")


def _bitget_granularity(tf: str, market: str) -> str:
    t = str(tf or "1h").strip().lower()
    if market == "spot":
        spot_map = {
            "1m": "1min",
            "3m": "3min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "6h": "6h",
            "12h": "12h",
            "1d": "1day",
        }
        if t in spot_map:
            return spot_map[t]
    mix_map = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "4h": "4H",
        "6h": "6H",
        "12h": "12H",
        "1d": "1D",
    }
    if t in mix_map:
        return mix_map[t]
    raise ValueError(f"unsupported_bitget_timeframe:{tf}")


def _parse_symbol_map(raw: str, *, upper_value: bool = True) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in str(raw).split(","):
        part = item.strip()
        if not part:
            continue
        if ":" in part:
            src, dst = part.split(":", 1)
            dst_norm = dst.strip().upper() if upper_value else dst.strip()
            out[src.strip().upper()] = dst_norm
            continue
        src = part.upper()
        dst = (src[:-4] if src.endswith("USDT") else src) if upper_value else part
        out[src] = dst
    return out


def _parse_dt_utc(raw: str) -> datetime:
    s = str(raw or "").strip()
    if not s:
        raise ValueError("empty_datetime")
    norm = s.replace(" ", "T")
    if norm.endswith("Z"):
        norm = norm[:-1] + "+00:00"
    dt_obj = datetime.fromisoformat(norm)
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    return dt_obj.astimezone(timezone.utc)


def _to_iso_z(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_from_ms(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)


def _iter_chunks_ms(start_ms: int, end_ms: int, chunk_days: int) -> Iterable[Tuple[int, int]]:
    cur = int(start_ms)
    chunk_ms = max(1, int(chunk_days)) * 24 * 60 * 60 * 1000
    while cur <= int(end_ms):
        nxt = min(int(end_ms), cur + chunk_ms - 1)
        yield cur, nxt
        cur = nxt + 1


def _chunk_id(idx: int, start_ms: int, end_ms: int) -> str:
    return f"{idx:05d}_{start_ms}-{end_ms}"


def _run_spec(
    *,
    market: str,
    timeframe: str,
    symbols: List[str],
    start_ms: int,
    end_ms: int,
    chunk_days: int,
) -> Dict[str, Any]:
    return {
        "market": str(market),
        "timeframe": str(timeframe),
        "symbols": list(symbols),
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "chunk_days": int(chunk_days),
    }


def _load_checkpoint(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError("invalid_checkpoint_not_object")
    return obj


def _save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(str(tmp), str(p))


def _save_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(str(tmp), str(p))


def _ensure_checkpoint(
    *,
    path: str,
    resume: bool,
    run_spec: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    now_iso = _to_iso_z(datetime.now(timezone.utc))
    if resume and os.path.exists(path):
        state = _load_checkpoint(path)
        ck_spec = dict(state.get("run_spec") or {})
        if ck_spec != run_spec:
            raise RuntimeError(
                json.dumps(
                    {
                        "checkpoint_mismatch": True,
                        "checkpoint_file": path,
                        "checkpoint_run_spec": ck_spec,
                        "current_run_spec": run_spec,
                    },
                    ensure_ascii=False,
                )
            )
        state["updated_at"] = now_iso
        if "completed_chunks" not in state or not isinstance(state.get("completed_chunks"), dict):
            state["completed_chunks"] = {}
        return state
    state = {
        "version": 1,
        "created_at": now_iso,
        "updated_at": now_iso,
        "run_spec": run_spec,
        "completed_chunks": {},
    }
    if (not dry_run) and path:
        _save_checkpoint(path, state)
    return state


def _build_session(proxy: str = "") -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    p = str(proxy or "").strip()
    if p:
        sess.proxies.update({"http": p, "https": p})
    return sess


def _is_rate_limit_error(err: str) -> bool:
    text = str(err or "").lower()
    return ("429" in text) or ("rate limit" in text) or ("too many requests" in text)


def _is_retryable_fetch_error(err: str) -> bool:
    text = str(err or "").lower()
    return bool(
        _is_rate_limit_error(text)
        or ("timeout" in text)
        or ("timed out" in text)
        or ("temporarily unavailable" in text)
        or ("503" in text)
        or ("504" in text)
        or ("502" in text)
        or ("connection reset" in text)
    )


def _fetch_bitget(
    sess: requests.Session,
    symbol: str,
    market: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    limit: int = 200,
) -> List[Bar]:
    cursor = int(start_ms)
    step_ms = _timeframe_minutes(timeframe) * 60 * 1000
    granularity = _bitget_granularity(timeframe, market)
    seen = set()
    out: List[Bar] = []
    while cursor <= end_ms:
        req_end = min(end_ms, cursor + limit * step_ms - 1)
        if market == "spot":
            params = {
                "symbol": symbol,
                "granularity": granularity,
                "startTime": str(cursor),
                "endTime": str(req_end),
                "limit": str(limit),
            }
            r = sess.get(BITGET_SPOT_URL, params=params, timeout=20)
        else:
            params = {
                "symbol": symbol,
                "productType": "USDT-FUTURES",
                "granularity": granularity,
                "startTime": str(cursor),
                "endTime": str(req_end),
                "limit": str(limit),
            }
            r = sess.get(BITGET_PERP_URL, params=params, timeout=20)
        r.raise_for_status()
        body = r.json()
        if str(body.get("code")) != "00000":
            raise RuntimeError(f"bitget_api_error market={market} symbol={symbol} body={body}")
        rows = body.get("data") or []
        if not rows:
            break
        batch: List[Bar] = []
        for row in rows:
            try:
                ts_ms = int(row[0])
                o = float(row[1])
                h = float(row[2])
                l = float(row[3])
                c = float(row[4])
                v = float(row[5])
            except Exception:
                continue
            if ts_ms < start_ms or ts_ms > end_ms or ts_ms in seen:
                continue
            seen.add(ts_ms)
            batch.append(Bar(ts_ms=ts_ms, open=o, high=h, low=l, close=c, volume=v))
        if not batch:
            break
        batch.sort(key=lambda x: x.ts_ms)
        out.extend(batch)
        cursor = batch[-1].ts_ms + step_ms
    out.sort(key=lambda x: x.ts_ms)
    return out


def _fetch_coingecko(sess: requests.Session, coin_id: str, days: int) -> List[Bar]:
    params = {
        "vs_currency": "usd",
        "days": str(max(1, int(days))),
        "interval": "hourly",
    }
    r = sess.get(COINGECKO_MARKET_CHART_URL.format(coin_id=coin_id), params=params, timeout=30)
    r.raise_for_status()
    body = r.json()
    prices = body.get("prices") or []
    vols = body.get("total_volumes") or []
    vol_map = {int(v[0]): float(v[1]) for v in vols if isinstance(v, list) and len(v) >= 2}
    bars: List[Bar] = []
    prev_price = None
    for row in prices:
        if not isinstance(row, list) or len(row) < 2:
            continue
        ts_ms = int(row[0])
        px = float(row[1])
        if px <= 0:
            continue
        o = float(prev_price if prev_price and prev_price > 0 else px)
        h = float(max(o, px))
        l = float(min(o, px))
        bars.append(
            Bar(
                ts_ms=ts_ms,
                open=o,
                high=h,
                low=l,
                close=float(px),
                volume=float(vol_map.get(ts_ms, 0.0)),
            )
        )
        prev_price = px
    bars.sort(key=lambda x: x.ts_ms)
    return bars


def _fetch_one_symbol(
    *,
    src_symbol: str,
    target_symbol: str,
    market: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    fallback_source: str,
    coingecko_map: Dict[str, str],
    proxy: str,
    source_label: str,
    days_for_fallback: int,
    max_retries: int,
    retry_backoff_sec: float,
) -> Tuple[str, List[Bar], str]:
    sess = _build_session(proxy=proxy)
    used_source = str(source_label)
    bars: List[Bar] | None = None
    last_err: Exception | None = None
    for attempt in range(1, max(1, int(max_retries)) + 1):
        try:
            bars = _fetch_bitget(
                sess,
                src_symbol,
                market,
                timeframe=str(timeframe),
                start_ms=start_ms,
                end_ms=end_ms,
            )
            break
        except Exception as exc:
            last_err = exc
            if attempt >= max(1, int(max_retries)):
                break
            sleep_s = min(90.0, float(retry_backoff_sec) * (2 ** (attempt - 1)) * (1.0 + random.uniform(0.0, 0.25)))
            time.sleep(max(0.1, sleep_s))
    if bars is None:
        if str(fallback_source) != "coingecko":
            if last_err is not None:
                raise last_err
            raise RuntimeError(f"bitget_fetch_failed symbol={src_symbol}")
        if str(timeframe).strip().lower() != "1h":
            raise RuntimeError("coingecko_fallback_only_supports_1h")
        coin_id = str(coingecko_map.get(src_symbol) or "").strip().lower()
        if not coin_id:
            raise RuntimeError(f"bitget_fetch_failed_and_no_coingecko_map symbol={src_symbol}")
        bars = _fetch_coingecko(sess, coin_id=coin_id, days=int(days_for_fallback))
        used_source = "coingecko_api"
    return target_symbol, bars, used_source


def _upsert_market_bars(
    conn: Any,
    target_symbol: str,
    timeframe: str,
    bars: List[Bar],
    source: str,
    *,
    page_size: int,
) -> int:
    if not bars:
        return 0
    values = [
        (
            target_symbol,
            timeframe,
            datetime.fromtimestamp(bar.ts_ms / 1000.0, tz=timezone.utc),
            float(bar.open),
            float(bar.high),
            float(bar.low),
            float(bar.close),
            float(bar.volume),
            0,
            source,
        )
        for bar in bars
    ]
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
                source = EXCLUDED.source
            """,
            values,
            template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            page_size=max(200, int(page_size)),
        )
    return len(values)


def _log_progress(event: str, **payload: object) -> None:
    body: Dict[str, object] = {
        "event": str(event),
        "ts": _to_iso_z(datetime.now(timezone.utc)),
    }
    for k, v in payload.items():
        body[str(k)] = v
    print(json.dumps(body, ensure_ascii=False), flush=True)


def _write_progress_file(path: str, payload: Dict[str, Any]) -> None:
    if not str(path or "").strip():
        return
    out = dict(payload)
    out["updated_at"] = _to_iso_z(datetime.now(timezone.utc))
    _save_json_atomic(path, out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest Bitget candles into market_bars (runtime path)")
    ap.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,AVAXUSDT,LINKUSDT",
        help="Bitget symbols",
    )
    ap.add_argument(
        "--symbol-map",
        default="BTCUSDT:BTC,ETHUSDT:ETH,SOLUSDT:SOL,BNBUSDT:BNB,XRPUSDT:XRP,ADAUSDT:ADA,DOGEUSDT:DOGE,TRXUSDT:TRX,AVAXUSDT:AVAX,LINKUSDT:LINK",
    )
    ap.add_argument("--market", choices=["perp", "spot"], default="perp")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--start", default="", help="inclusive UTC start time, e.g. 2025-01-01T00:00:00Z")
    ap.add_argument("--end", default="", help="inclusive UTC end time, default now")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--fallback-source", choices=["none", "coingecko"], default="coingecko")
    ap.add_argument(
        "--coingecko-map",
        default="BTCUSDT:bitcoin,ETHUSDT:ethereum,SOLUSDT:solana,BNBUSDT:binancecoin,XRPUSDT:ripple,ADAUSDT:cardano,DOGEUSDT:dogecoin,TRXUSDT:tron,AVAXUSDT:avalanche-2,LINKUSDT:chainlink",
    )
    ap.add_argument("--proxy", default=os.getenv("HTTPS_PROXY", os.getenv("ALL_PROXY", "")))
    ap.add_argument("--out-csv", default="", help="optional csv output path")
    ap.add_argument("--skip-db", action="store_true", help="fetch only, do not write database")
    ap.add_argument("--database-url", default=DATABASE_URL)
    ap.add_argument("--source", default="bitget_api")
    ap.add_argument("--workers", type=int, default=max(1, int(os.getenv("INGEST_WORKERS", "4"))))
    ap.add_argument("--min-workers", type=int, default=max(1, int(os.getenv("INGEST_MIN_WORKERS", "1"))))
    ap.add_argument("--max-workers", type=int, default=max(0, int(os.getenv("INGEST_MAX_WORKERS", "0"))))
    ap.add_argument("--adaptive-workers", type=int, default=int(os.getenv("INGEST_ADAPTIVE_WORKERS", "1")))
    ap.add_argument("--rate-limit-backoff-sec", type=float, default=float(os.getenv("INGEST_RATE_LIMIT_BACKOFF_SEC", "5.0")))
    ap.add_argument("--max-rate-limit-backoff-sec", type=float, default=float(os.getenv("INGEST_MAX_RATE_LIMIT_BACKOFF_SEC", "120.0")))
    ap.add_argument("--max-fetch-retries", type=int, default=max(1, int(os.getenv("INGEST_MAX_FETCH_RETRIES", "4"))))
    ap.add_argument("--retry-backoff-sec", type=float, default=float(os.getenv("INGEST_RETRY_BACKOFF_SEC", "1.0")))
    ap.add_argument("--chunk-days", type=int, default=max(1, int(os.getenv("INGEST_CHUNK_DAYS", "30"))))
    ap.add_argument("--min-split-chunk-days", type=float, default=float(os.getenv("INGEST_MIN_SPLIT_CHUNK_DAYS", "2.0")))
    ap.add_argument("--split-failed-symbols", type=int, default=int(os.getenv("INGEST_SPLIT_FAILED_SYMBOLS", "1")))
    ap.add_argument("--db-page-size", type=int, default=max(200, int(os.getenv("INGEST_DB_PAGE_SIZE", "1200"))))
    ap.add_argument("--db-commit-every-symbols", type=int, default=max(1, int(os.getenv("INGEST_DB_COMMIT_EVERY_SYMBOLS", "4"))))
    ap.add_argument("--checkpoint-file", default="", help="json checkpoint file for resumable long-range backfill")
    ap.add_argument("--progress-file", default=os.getenv("INGEST_PROGRESS_FILE", ""))
    ap.add_argument("--resume", action="store_true", help="resume from checkpoint-file")
    ap.add_argument("--max-chunks", type=int, default=0, help="optional chunk cap for staged backfill")
    ap.add_argument("--dry-run", action="store_true", help="plan chunks only; no network/db/csv writes")
    ap.add_argument("--allow-partial", action="store_true", help="continue if some symbols fail")
    args = ap.parse_args()

    end_dt = _parse_dt_utc(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    if str(args.start).strip():
        start_dt = _parse_dt_utc(args.start)
    else:
        start_dt = end_dt - timedelta(days=max(1, int(args.days)))
    if end_dt <= start_dt:
        raise RuntimeError("invalid_time_range_end_must_be_gt_start")
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    symbol_map = _parse_symbol_map(args.symbol_map, upper_value=True)
    coingecko_map = _parse_symbol_map(args.coingecko_map, upper_value=False)
    req_symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    run_spec = _run_spec(
        market=str(args.market),
        timeframe=str(args.timeframe),
        symbols=req_symbols,
        start_ms=start_ms,
        end_ms=end_ms,
        chunk_days=max(1, int(args.chunk_days)),
    )
    checkpoint_file = str(args.checkpoint_file).strip()
    progress_file = str(args.progress_file).strip()
    if (not progress_file) and checkpoint_file:
        progress_file = f"{checkpoint_file}.progress.json"
    checkpoint = None
    completed_chunk_ids = set()
    if checkpoint_file:
        checkpoint = _ensure_checkpoint(
            path=checkpoint_file,
            resume=bool(args.resume),
            run_spec=run_spec,
            dry_run=bool(args.dry_run),
        )
        completed_chunk_ids = set((checkpoint.get("completed_chunks") or {}).keys())

    summary_rows: Dict[str, int] = {}
    summary_sources: Dict[str, set] = {}
    out_csv = str(args.out_csv).strip()
    csv_file = None
    csv_writer = None
    if out_csv and not bool(args.dry_run):
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        csv_file = open(out_csv, "w", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["symbol", "timeframe", "ts", "open", "high", "low", "close", "volume", "trades_count", "source"],
        )
        csv_writer.writeheader()

    failed: List[Dict[str, str]] = []
    chunks = list(_iter_chunks_ms(start_ms, end_ms, chunk_days=max(1, int(args.chunk_days))))
    dry_plan: List[Dict[str, object]] = []
    chunks_attempted = 0
    chunks_skipped_resume = 0
    chunks_completed = 0
    workers = max(1, int(args.workers))
    min_workers = max(1, int(args.min_workers))
    max_workers = max(workers, int(args.max_workers or workers))
    current_workers = min(max_workers, max(min_workers, workers))
    adaptive_workers = bool(int(args.adaptive_workers))
    backoff_sec = 0.0
    no_error_streak = 0
    run_started = time.monotonic()
    _log_progress(
        "ingest_start",
        market=args.market,
        timeframe=args.timeframe,
        symbols=len(req_symbols),
        start=_to_iso_z(start_dt),
        end=_to_iso_z(end_dt),
        chunks_total=len(chunks),
        checkpoint_file=checkpoint_file or None,
    )
    _write_progress_file(
        progress_file,
        {
            "status": "running",
            "market": str(args.market),
            "timeframe": str(args.timeframe),
            "start": _to_iso_z(start_dt),
            "end": _to_iso_z(end_dt),
            "chunks_total": int(len(chunks)),
            "chunks_completed": int(chunks_completed),
            "chunks_attempted": int(chunks_attempted),
            "chunks_skipped_resume": int(chunks_skipped_resume),
            "workers": int(current_workers),
            "checkpoint_file": checkpoint_file or None,
        },
    )
    try:
        for idx, (chunk_start_ms, chunk_end_ms) in enumerate(chunks):
            if int(args.max_chunks) > 0 and chunks_attempted >= int(args.max_chunks):
                break
            # Bitget may reject zero-width ranges; skip degenerate chunk.
            if int(chunk_end_ms) <= int(chunk_start_ms):
                continue
            chunk_name = _chunk_id(idx, chunk_start_ms, chunk_end_ms)
            if chunk_name in completed_chunk_ids:
                chunks_skipped_resume += 1
                continue
            chunks_attempted += 1
            chunk_started = time.monotonic()
            _log_progress(
                "chunk_start",
                chunk=chunk_name,
                chunk_index=int(idx + 1),
                chunks_total=len(chunks),
                start=_to_iso_z(_utc_from_ms(chunk_start_ms)),
                end=_to_iso_z(_utc_from_ms(chunk_end_ms)),
                workers=int(current_workers),
            )
            _write_progress_file(
                progress_file,
                {
                    "status": "running",
                    "market": str(args.market),
                    "timeframe": str(args.timeframe),
                    "chunk": chunk_name,
                    "chunk_index": int(idx + 1),
                    "chunks_total": int(len(chunks)),
                    "chunks_completed": int(chunks_completed),
                    "chunks_attempted": int(chunks_attempted),
                    "chunks_skipped_resume": int(chunks_skipped_resume),
                    "workers": int(current_workers),
                    "checkpoint_file": checkpoint_file or None,
                },
            )
            if bool(args.dry_run):
                dry_plan.append(
                    {
                        "chunk": chunk_name,
                        "start": _to_iso_z(_utc_from_ms(chunk_start_ms)),
                        "end": _to_iso_z(_utc_from_ms(chunk_end_ms)),
                        "symbols": req_symbols,
                    }
                )
                continue

            chunk_failed: List[Dict[str, str]] = []
            chunk_rows = 0
            days_for_fallback = max(1, int(((chunk_end_ms - chunk_start_ms) / (24 * 60 * 60 * 1000)) + 2))
            failed_by_symbol: Dict[str, Dict[str, str]] = {}
            split_recovered = 0
            db_conn = psycopg2.connect(args.database_url) if not bool(args.skip_db) else None
            symbols_since_commit = 0
            try:
                with ThreadPoolExecutor(max_workers=int(current_workers)) as ex:
                    futs = {}
                    for src_symbol in req_symbols:
                        target = symbol_map.get(src_symbol) or (src_symbol[:-4] if src_symbol.endswith("USDT") else src_symbol)
                        fut = ex.submit(
                            _fetch_one_symbol,
                            src_symbol=src_symbol,
                            target_symbol=target,
                            market=str(args.market),
                            timeframe=str(args.timeframe),
                            start_ms=chunk_start_ms,
                            end_ms=chunk_end_ms,
                            fallback_source=str(args.fallback_source),
                            coingecko_map=coingecko_map,
                            proxy=str(args.proxy),
                            source_label=str(args.source),
                            days_for_fallback=days_for_fallback,
                            max_retries=int(args.max_fetch_retries),
                            retry_backoff_sec=float(args.retry_backoff_sec),
                        )
                        futs[fut] = {"src_symbol": src_symbol, "target_symbol": target}
                    for fut in as_completed(futs):
                        meta = futs[fut]
                        src_symbol = str(meta.get("src_symbol") or "")
                        target = str(meta.get("target_symbol") or "")
                        try:
                            target, bars, used_source = fut.result()
                            if csv_writer is not None:
                                for bar in bars:
                                    csv_writer.writerow(
                                        {
                                            "symbol": target,
                                            "timeframe": str(args.timeframe),
                                            "ts": datetime.fromtimestamp(bar.ts_ms / 1000.0, tz=timezone.utc).isoformat(),
                                            "open": float(bar.open),
                                            "high": float(bar.high),
                                            "low": float(bar.low),
                                            "close": float(bar.close),
                                            "volume": float(bar.volume),
                                            "trades_count": 0,
                                            "source": used_source,
                                        }
                                    )
                            if bool(args.skip_db):
                                count = int(len(bars))
                            else:
                                assert db_conn is not None
                                count = _upsert_market_bars(
                                    db_conn,
                                    target_symbol=target,
                                    timeframe=str(args.timeframe),
                                    bars=bars,
                                    source=used_source,
                                    page_size=int(args.db_page_size),
                                )
                                symbols_since_commit += 1
                                if symbols_since_commit >= int(args.db_commit_every_symbols):
                                    db_conn.commit()
                                    symbols_since_commit = 0
                            chunk_rows += int(count)
                            summary_rows[target] = int(summary_rows.get(target, 0)) + int(count)
                            summary_sources.setdefault(target, set()).add(str(used_source))
                        except Exception as exc:
                            row = {"src_symbol": src_symbol, "target_symbol": target, "error": str(exc)}
                            failed_by_symbol[src_symbol] = row
                            chunk_failed.append(row)
                            failed.append({"chunk": chunk_name, **row})
                            summary_rows.setdefault(target or src_symbol, 0)
                            summary_sources.setdefault(target or src_symbol, set()).add("failed")

                window_days = float(max(0.01, (chunk_end_ms - chunk_start_ms + 1) / (24 * 60 * 60 * 1000)))
                allow_split = bool(int(args.split_failed_symbols)) and (window_days >= float(args.min_split_chunk_days))
                if allow_split and failed_by_symbol:
                    split_start_1 = int(chunk_start_ms)
                    split_end_1 = int((chunk_start_ms + chunk_end_ms) // 2)
                    split_start_2 = int(split_end_1 + 1)
                    split_end_2 = int(chunk_end_ms)
                    recovered_symbols: List[str] = []
                    for src_symbol, row in list(failed_by_symbol.items()):
                        target = str(row.get("target_symbol") or "")
                        err_msg = str(row.get("error") or "")
                        if not _is_retryable_fetch_error(err_msg):
                            continue
                        try:
                            _, bars_a, src_a = _fetch_one_symbol(
                                src_symbol=src_symbol,
                                target_symbol=target,
                                market=str(args.market),
                                timeframe=str(args.timeframe),
                                start_ms=split_start_1,
                                end_ms=split_end_1,
                                fallback_source=str(args.fallback_source),
                                coingecko_map=coingecko_map,
                                proxy=str(args.proxy),
                                source_label=str(args.source),
                                days_for_fallback=max(1, int(((split_end_1 - split_start_1) / (24 * 60 * 60 * 1000)) + 2)),
                                max_retries=int(args.max_fetch_retries),
                                retry_backoff_sec=float(args.retry_backoff_sec),
                            )
                            _, bars_b, src_b = _fetch_one_symbol(
                                src_symbol=src_symbol,
                                target_symbol=target,
                                market=str(args.market),
                                timeframe=str(args.timeframe),
                                start_ms=split_start_2,
                                end_ms=split_end_2,
                                fallback_source=str(args.fallback_source),
                                coingecko_map=coingecko_map,
                                proxy=str(args.proxy),
                                source_label=str(args.source),
                                days_for_fallback=max(1, int(((split_end_2 - split_start_2) / (24 * 60 * 60 * 1000)) + 2)),
                                max_retries=int(args.max_fetch_retries),
                                retry_backoff_sec=float(args.retry_backoff_sec),
                            )
                            merged: Dict[int, Bar] = {}
                            for bar in list(bars_a) + list(bars_b):
                                merged[int(bar.ts_ms)] = bar
                            bars = [merged[k] for k in sorted(merged.keys())]
                            if csv_writer is not None:
                                split_src = src_a if src_a == src_b else "split_mixed"
                                for bar in bars:
                                    csv_writer.writerow(
                                        {
                                            "symbol": target,
                                            "timeframe": str(args.timeframe),
                                            "ts": datetime.fromtimestamp(bar.ts_ms / 1000.0, tz=timezone.utc).isoformat(),
                                            "open": float(bar.open),
                                            "high": float(bar.high),
                                            "low": float(bar.low),
                                            "close": float(bar.close),
                                            "volume": float(bar.volume),
                                            "trades_count": 0,
                                            "source": split_src,
                                        }
                                    )
                            if bool(args.skip_db):
                                count = int(len(bars))
                            else:
                                assert db_conn is not None
                                split_src = src_a if src_a == src_b else "split_mixed"
                                count = _upsert_market_bars(
                                    db_conn,
                                    target_symbol=target,
                                    timeframe=str(args.timeframe),
                                    bars=bars,
                                    source=split_src,
                                    page_size=int(args.db_page_size),
                                )
                                symbols_since_commit += 1
                                if symbols_since_commit >= int(args.db_commit_every_symbols):
                                    db_conn.commit()
                                    symbols_since_commit = 0
                            chunk_rows += int(count)
                            summary_rows[target] = int(summary_rows.get(target, 0)) + int(count)
                            summary_sources.setdefault(target, set()).add("split_recovered")
                            split_recovered += 1
                            recovered_symbols.append(src_symbol)
                            _log_progress(
                                "chunk_split_recover",
                                chunk=chunk_name,
                                symbol=src_symbol,
                                rows=int(count),
                                window_days=window_days,
                            )
                        except Exception as split_exc:
                            failed_by_symbol[src_symbol] = {
                                "src_symbol": src_symbol,
                                "target_symbol": target,
                                "error": f"{err_msg} | split_retry_failed:{split_exc}",
                            }
                    if recovered_symbols:
                        survived = [r for r in chunk_failed if str(r.get("src_symbol") or "") not in set(recovered_symbols)]
                        chunk_failed = survived
                        for src_symbol in recovered_symbols:
                            failed_by_symbol.pop(src_symbol, None)
                        failed = [
                            r
                            for r in failed
                            if not (str(r.get("chunk") or "") == chunk_name and str(r.get("src_symbol") or "") in set(recovered_symbols))
                        ]
                if db_conn is not None:
                    db_conn.commit()
            except Exception:
                if db_conn is not None:
                    db_conn.rollback()
                raise
            finally:
                if db_conn is not None:
                    db_conn.close()

            if checkpoint_file and checkpoint is not None:
                completed_map = checkpoint.setdefault("completed_chunks", {})
                completed_map[chunk_name] = {
                    "start": _to_iso_z(_utc_from_ms(chunk_start_ms)),
                    "end": _to_iso_z(_utc_from_ms(chunk_end_ms)),
                    "rows": int(chunk_rows),
                    "failed_symbols": chunk_failed,
                    "completed_at": _to_iso_z(datetime.now(timezone.utc)),
                }
                checkpoint["updated_at"] = _to_iso_z(datetime.now(timezone.utc))
                _save_checkpoint(checkpoint_file, checkpoint)
                completed_chunk_ids.add(chunk_name)

            chunks_completed += 1
            _log_progress(
                "chunk_done",
                chunk=chunk_name,
                rows=int(chunk_rows),
                failed_symbols=len(chunk_failed),
                split_recovered=int(split_recovered),
                chunks_completed=int(chunks_completed),
                chunks_attempted=int(chunks_attempted),
                chunks_total=len(chunks),
                workers=int(current_workers),
                elapsed_sec=round(float(time.monotonic() - chunk_started), 3),
            )
            rate_limit_hits = sum(1 for row in chunk_failed if _is_rate_limit_error(str(row.get("error") or "")))
            retryable_hits = sum(1 for row in chunk_failed if _is_retryable_fetch_error(str(row.get("error") or "")))
            if adaptive_workers:
                if rate_limit_hits > 0 or retryable_hits > 0:
                    old_workers = int(current_workers)
                    current_workers = max(min_workers, current_workers - 1)
                    no_error_streak = 0
                    backoff_sec = min(float(args.max_rate_limit_backoff_sec), max(float(args.rate_limit_backoff_sec), (backoff_sec * 1.6) + 1.0))
                    if old_workers != current_workers:
                        _log_progress(
                            "workers_downshift",
                            old_workers=old_workers,
                            new_workers=int(current_workers),
                            rate_limit_hits=int(rate_limit_hits),
                            retryable_hits=int(retryable_hits),
                            chunk=chunk_name,
                        )
                    _log_progress("rate_limit_backoff", sleep_sec=round(float(backoff_sec), 3), chunk=chunk_name)
                    time.sleep(max(0.0, float(backoff_sec)))
                else:
                    no_error_streak += 1
                    if no_error_streak >= 3 and current_workers < max_workers:
                        old_workers = int(current_workers)
                        current_workers += 1
                        no_error_streak = 0
                        backoff_sec = max(0.0, backoff_sec * 0.5)
                        _log_progress("workers_upshift", old_workers=old_workers, new_workers=int(current_workers), chunk=chunk_name)
            _write_progress_file(
                progress_file,
                {
                    "status": "running",
                    "market": str(args.market),
                    "timeframe": str(args.timeframe),
                    "chunk": chunk_name,
                    "chunks_total": int(len(chunks)),
                    "chunks_completed": int(chunks_completed),
                    "chunks_attempted": int(chunks_attempted),
                    "chunks_skipped_resume": int(chunks_skipped_resume),
                    "rows_last_chunk": int(chunk_rows),
                    "failed_last_chunk": int(len(chunk_failed)),
                    "split_recovered_last_chunk": int(split_recovered),
                    "workers": int(current_workers),
                    "checkpoint_file": checkpoint_file or None,
                },
            )
            if chunk_failed and not bool(args.allow_partial):
                raise RuntimeError(json.dumps({"failed_symbols": chunk_failed, "chunk": chunk_name}, ensure_ascii=False))
    finally:
        if csv_file is not None:
            csv_file.close()

    if bool(args.dry_run):
        _write_progress_file(
            progress_file,
            {
                "status": "dry_run",
                "market": str(args.market),
                "timeframe": str(args.timeframe),
                "chunks_total": int(len(chunks)),
                "chunks_completed": int(chunks_completed),
                "chunks_attempted": int(chunks_attempted),
                "chunks_skipped_resume": int(chunks_skipped_resume),
                "workers": int(current_workers),
                "checkpoint_file": checkpoint_file or None,
            },
        )
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "market": args.market,
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                    "chunks_total": len(chunks),
                    "chunks_planned": len(dry_plan),
                    "chunks_skipped_resume": chunks_skipped_resume,
                    "chunk_days": int(args.chunk_days),
                    "checkpoint_file": checkpoint_file or None,
                    "plan": dry_plan,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if failed and not bool(args.allow_partial):
        _write_progress_file(
            progress_file,
            {
                "status": "failed",
                "market": str(args.market),
                "timeframe": str(args.timeframe),
                "chunks_total": int(len(chunks)),
                "chunks_completed": int(chunks_completed),
                "chunks_attempted": int(chunks_attempted),
                "chunks_skipped_resume": int(chunks_skipped_resume),
                "workers": int(current_workers),
                "checkpoint_file": checkpoint_file or None,
                "failed_count": int(len(failed)),
                "error": "failed_symbols_present_and_allow_partial_disabled",
            },
        )
        raise RuntimeError(json.dumps({"failed_symbols": failed}, ensure_ascii=False))

    summary = {
        k: {"rows": int(summary_rows.get(k, 0)), "sources": sorted(list(summary_sources.get(k, set())))}
        for k in sorted(summary_rows.keys())
    }

    print(
        json.dumps(
            {
                "status": "ok",
                "market": args.market,
                "days": int(args.days),
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "chunks_total": len(chunks),
                "chunks_attempted": chunks_attempted,
                "chunks_completed": chunks_completed,
                "chunks_skipped_resume": chunks_skipped_resume,
                "chunk_days": int(args.chunk_days),
                "checkpoint_file": checkpoint_file or None,
                "failed": failed,
                "inserted": summary,
                "elapsed_sec": round(float(time.monotonic() - run_started), 3),
            },
            ensure_ascii=False,
        )
    )
    _write_progress_file(
        progress_file,
        {
            "status": "completed",
            "market": str(args.market),
            "timeframe": str(args.timeframe),
            "chunks_total": int(len(chunks)),
            "chunks_completed": int(chunks_completed),
            "chunks_attempted": int(chunks_attempted),
            "chunks_skipped_resume": int(chunks_skipped_resume),
            "workers": int(current_workers),
            "checkpoint_file": checkpoint_file or None,
            "failed_count": int(len(failed)),
            "elapsed_sec": round(float(time.monotonic() - run_started), 3),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
