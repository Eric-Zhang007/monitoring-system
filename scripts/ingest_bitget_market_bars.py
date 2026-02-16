#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import random
import time
from typing import Dict, List, Tuple

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


def _upsert_market_bars(db_url: str, target_symbol: str, timeframe: str, bars: List[Bar], source: str) -> int:
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
    with psycopg2.connect(db_url) as conn:
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
                page_size=500,
            )
    return len(values)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest Bitget candles into market_bars (no-docker path)")
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
    ap.add_argument("--max-fetch-retries", type=int, default=max(1, int(os.getenv("INGEST_MAX_FETCH_RETRIES", "4"))))
    ap.add_argument("--retry-backoff-sec", type=float, default=float(os.getenv("INGEST_RETRY_BACKOFF_SEC", "1.0")))
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
    summary: Dict[str, Dict[str, object]] = {}
    out_csv = str(args.out_csv).strip()
    csv_file = None
    csv_writer = None
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        csv_file = open(out_csv, "w", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["symbol", "timeframe", "ts", "open", "high", "low", "close", "volume", "trades_count", "source"],
        )
        csv_writer.writeheader()

    failed: List[Dict[str, str]] = []
    workers = max(1, int(args.workers))
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for src_symbol in req_symbols:
                target = symbol_map.get(src_symbol) or (src_symbol[:-4] if src_symbol.endswith("USDT") else src_symbol)
                fut = ex.submit(
                    _fetch_one_symbol,
                    src_symbol=src_symbol,
                    target_symbol=target,
                    market=str(args.market),
                    timeframe=str(args.timeframe),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    fallback_source=str(args.fallback_source),
                    coingecko_map=coingecko_map,
                    proxy=str(args.proxy),
                    source_label=str(args.source),
                    days_for_fallback=int(args.days),
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
                        count = _upsert_market_bars(args.database_url, target_symbol=target, timeframe=str(args.timeframe), bars=bars, source=used_source)
                    summary[target] = {"rows": int(count), "source": used_source}
                except Exception as exc:
                    failed.append({"src_symbol": src_symbol, "target_symbol": target, "error": str(exc)})
                    summary[target or src_symbol] = {"rows": 0, "source": "failed", "error": str(exc)}
    finally:
        if csv_file is not None:
            csv_file.close()

    if failed and not bool(args.allow_partial):
        raise RuntimeError(json.dumps({"failed_symbols": failed}, ensure_ascii=False))

    print(
        json.dumps(
            {
                "status": "ok",
                "market": args.market,
                "days": int(args.days),
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "failed": failed,
                "inserted": summary,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
