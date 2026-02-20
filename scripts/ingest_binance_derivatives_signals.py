#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2.extras import execute_values
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
try:
    from inference.liquid_feature_contract import DERIVATIVE_METRIC_NAMES as CONTRACT_DERIVATIVE_METRICS
except Exception:
    CONTRACT_DERIVATIVE_METRICS = [
        "long_short_ratio_global_accounts",
        "long_short_ratio_top_accounts",
        "long_short_ratio_top_positions",
        "taker_buy_sell_ratio",
        "basis_rate",
        "annualized_basis_rate",
    ]

BINANCE_FAPI_BASE = "https://fapi.binance.com"

METRIC_ENDPOINTS: Dict[str, Dict[str, Any]] = {
    "long_short_ratio_global_accounts": {
        "path": "/futures/data/globalLongShortAccountRatio",
        "value_field": "longShortRatio",
        "kind": "symbol",
    },
    "long_short_ratio_top_accounts": {
        "path": "/futures/data/topLongShortAccountRatio",
        "value_field": "longShortRatio",
        "kind": "symbol",
    },
    "long_short_ratio_top_positions": {
        "path": "/futures/data/topLongShortPositionRatio",
        "value_field": "longShortRatio",
        "kind": "symbol",
    },
    "taker_buy_sell_ratio": {
        "path": "/futures/data/takerlongshortRatio",
        "value_field": "buySellRatio",
        "kind": "symbol",
    },
    "basis_rate": {
        "path": "/futures/data/basis",
        "value_field": "basisRate",
        "kind": "pair",
    },
    "annualized_basis_rate": {
        "path": "/futures/data/basis",
        "value_field": "annualizedBasisRate",
        "kind": "pair",
    },
}


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


def _parse_symbols(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for piece in str(raw or "").split(","):
        sym = piece.strip().upper().replace("$", "")
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def _parse_symbol_map(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in str(raw or "").split(","):
        token = item.strip()
        if not token:
            continue
        if ":" in token:
            left, right = token.split(":", 1)
            sym = left.strip().upper()
            pair = right.strip().upper()
        else:
            sym = token.upper()
            pair = f"{sym}USDT"
        if sym:
            out[sym] = pair
    return out


def _safe_float(raw: Any) -> Optional[float]:
    try:
        return float(raw)
    except Exception:
        return None


def _period_ms(period: str) -> int:
    text = str(period or "").strip().lower()
    if text.endswith("m"):
        return max(1, int(text[:-1] or "1")) * 60_000
    if text.endswith("h"):
        return max(1, int(text[:-1] or "1")) * 3_600_000
    if text.endswith("d"):
        return max(1, int(text[:-1] or "1")) * 86_400_000
    raise ValueError(f"unsupported_period:{period}")


def _build_session(proxy: str = "") -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.7,
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


def _chunk_rows(rows: Sequence[Tuple[Any, ...]], size: int) -> Iterable[List[Tuple[Any, ...]]]:
    batch: List[Tuple[Any, ...]] = []
    for row in rows:
        batch.append(tuple(row))
        if len(batch) >= int(size):
            yield batch
            batch = []
    if batch:
        yield batch


def _dedup_points(points: Sequence[Tuple[int, float]]) -> List[Tuple[int, float]]:
    dedup: Dict[int, float] = {}
    for ts_ms, v in points:
        dedup[int(ts_ms)] = float(v)
    return sorted(dedup.items(), key=lambda x: x[0])


def _expand_forward_fill(
    points: Sequence[Tuple[int, float]],
    *,
    start_ms: int,
    end_ms: int,
    bucket_ms: int,
) -> List[Tuple[int, float]]:
    if not points:
        return []
    if bucket_ms <= 0:
        return _dedup_points(points)
    base = _dedup_points(points)
    if base and int(base[0][0]) > int(start_ms):
        base = [(int(start_ms), float(base[0][1]))] + base
    out: List[Tuple[int, float]] = []
    clipped_start = int(start_ms)
    clipped_end = int(end_ms)
    for idx, (ts_ms, value) in enumerate(base):
        right = base[idx + 1][0] - 1 if idx + 1 < len(base) else clipped_end
        seg_start = max(int(ts_ms), clipped_start)
        seg_end = min(int(right), clipped_end)
        if seg_end < seg_start:
            continue
        bucket_start = int((seg_start // bucket_ms) * bucket_ms)
        if bucket_start < clipped_start:
            bucket_start += bucket_ms
        cur = bucket_start
        while cur <= seg_end:
            out.append((int(cur), float(value)))
            cur += bucket_ms
    return _dedup_points(out)


def _fetch_metric_series(
    session: requests.Session,
    *,
    path: str,
    value_field: str,
    kind: str,
    pair: str,
    period: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    max_pages: int,
    sleep_sec: float,
    contract_type: str,
) -> Tuple[List[Tuple[int, float]], Optional[str]]:
    url = f"{BINANCE_FAPI_BASE}{str(path)}"
    rows: List[Tuple[int, float]] = []
    cursor = int(start_ms)
    warn: Optional[str] = None
    pages = 0

    while cursor <= int(end_ms):
        params: Dict[str, Any] = {
            "period": str(period),
            "startTime": int(cursor),
            "endTime": int(end_ms),
            "limit": int(max(1, min(500, int(limit)))),
        }
        if str(kind) == "pair":
            params["pair"] = str(pair)
            params["contractType"] = str(contract_type)
        else:
            params["symbol"] = str(pair)

        resp = session.get(url, params=params, timeout=25)
        if resp.status_code >= 400:
            msg = ""
            try:
                body_err = resp.json()
                msg = str(body_err.get("msg") or body_err)
            except Exception:
                msg = (resp.text or "")[:200]
            if "startTime" in msg or "endTime" in msg:
                warn = f"window_limited:{msg}"
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
            if ts_ms <= 0 or ts_ms < int(start_ms) or ts_ms > int(end_ms):
                continue
            value = _safe_float(item.get(str(value_field)))
            if value is None:
                continue
            rows.append((int(ts_ms), float(value)))
            if ts_ms > max_seen:
                max_seen = ts_ms

        if max_seen <= cursor:
            break
        cursor = int(max_seen + 1)
        pages += 1
        if int(max_pages) > 0 and pages >= int(max_pages):
            break
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))

    return _dedup_points(rows), warn


def _replace_window(
    conn,
    *,
    symbol: str,
    chain: str,
    source: str,
    metric_names: Sequence[str],
    start_dt: datetime,
    end_dt: datetime,
) -> int:
    if not metric_names:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM onchain_signals
            WHERE asset_symbol = %s
              AND chain = %s
              AND source = %s
              AND metric_name = ANY(%s)
              AND ts >= %s
              AND ts <= %s
            """,
            (
                str(symbol).upper(),
                str(chain),
                str(source),
                list(metric_names),
                start_dt,
                end_dt,
            ),
        )
        return int(cur.rowcount or 0)


def _insert_series(
    conn,
    *,
    symbol: str,
    chain: str,
    source: str,
    metric_name: str,
    series: Sequence[Tuple[int, float]],
    batch_size: int,
) -> int:
    if not series:
        return 0
    inserted = 0
    with conn.cursor() as cur:
        for batch in _chunk_rows(series, size=max(200, int(batch_size))):
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest Binance derivatives regime signals into onchain_signals")
    ap.add_argument("--database-url", default=DATABASE_URL)
    ap.add_argument("--symbols", default="BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK")
    ap.add_argument(
        "--symbol-map",
        default="BTC:BTCUSDT,ETH:ETHUSDT,SOL:SOLUSDT,BNB:BNBUSDT,XRP:XRPUSDT,ADA:ADAUSDT,DOGE:DOGEUSDT,TRX:TRXUSDT,AVAX:AVAXUSDT,LINK:LINKUSDT",
    )
    ap.add_argument("--start", default="2018-01-01T00:00:00Z")
    ap.add_argument("--end", default="")
    ap.add_argument("--period", default="5m")
    ap.add_argument("--contract-type", default="PERPETUAL")
    ap.add_argument("--max-lookback-days", type=int, default=int(os.getenv("BINANCE_DERIV_MAX_LOOKBACK_DAYS", "30")))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--sleep-sec", type=float, default=0.08)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--chain", default="derivatives")
    ap.add_argument("--source", default="binance_fapi_derivatives")
    ap.add_argument(
        "--metrics",
        default=",".join(CONTRACT_DERIVATIVE_METRICS),
    )
    ap.add_argument("--replace-window", action="store_true", default=True)
    ap.add_argument("--no-replace-window", dest="replace_window", action="store_false")
    ap.add_argument("--proxy", default=os.getenv("HTTPS_PROXY", os.getenv("ALL_PROXY", "")))
    ap.add_argument("--expand-to-period", action="store_true", default=True)
    ap.add_argument("--no-expand-to-period", dest="expand_to_period", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise RuntimeError("empty_symbols")
    symbol_map = _parse_symbol_map(args.symbol_map)

    selected_metrics: List[str] = []
    seen = set()
    for item in str(args.metrics or "").split(","):
        m = str(item).strip()
        if not m or m in seen:
            continue
        if m not in METRIC_ENDPOINTS:
            continue
        seen.add(m)
        selected_metrics.append(m)
    if not selected_metrics:
        raise RuntimeError("empty_metrics")

    start_dt = _parse_dt_utc(args.start)
    end_dt = _parse_dt_utc(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    if end_dt <= start_dt:
        raise RuntimeError("invalid_time_range")

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    floor_dt = datetime.now(timezone.utc) - timedelta(days=max(1, int(args.max_lookback_days)))
    floor_ms = int(floor_dt.timestamp() * 1000)
    clamped_start_ms = max(start_ms, floor_ms)
    bucket_ms = _period_ms(str(args.period))

    out: Dict[str, Any] = {
        "status": "ok",
        "start": _to_iso_z(start_dt),
        "end": _to_iso_z(end_dt),
        "requested_metrics": selected_metrics,
        "period": str(args.period),
        "contract_type": str(args.contract_type),
        "chain": str(args.chain),
        "source": str(args.source),
        "replace_window": bool(args.replace_window),
        "dry_run": bool(args.dry_run),
        "symbols": symbols,
        "warnings": [],
        "per_symbol": {},
    }

    if clamped_start_ms > start_ms:
        out["warnings"].append(
            f"start clamped to {_to_iso_z(_to_dt_utc(clamped_start_ms))} due to exchange retention limit"
        )

    sess = _build_session(proxy=str(args.proxy))

    if bool(args.dry_run):
        for sym in symbols:
            pair = symbol_map.get(sym, f"{sym}USDT")
            out["per_symbol"][sym] = {
                "pair": pair,
                "fetch_start": _to_iso_z(_to_dt_utc(clamped_start_ms)),
                "fetch_end": _to_iso_z(_to_dt_utc(end_ms)),
                "metrics": selected_metrics,
            }
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
            sym_obj: Dict[str, Any] = {
                "pair": pair,
                "fetch_start": _to_iso_z(_to_dt_utc(clamped_start_ms)),
                "fetch_end": _to_iso_z(_to_dt_utc(end_ms)),
                "metrics": {},
                "warnings": [],
            }
            basis_series_for_annualized: List[Tuple[int, float]] = []
            if bool(args.replace_window):
                deleted = _replace_window(
                    conn,
                    symbol=sym,
                    chain=str(args.chain),
                    source=str(args.source),
                    metric_names=selected_metrics,
                    start_dt=start_dt,
                    end_dt=end_dt,
                )
                sym_obj["deleted_rows"] = int(deleted)

            for metric_name in selected_metrics:
                conf = METRIC_ENDPOINTS.get(metric_name) or {}
                series, warn = _fetch_metric_series(
                    sess,
                    path=str(conf.get("path") or ""),
                    value_field=str(conf.get("value_field") or ""),
                    kind=str(conf.get("kind") or "symbol"),
                    pair=pair,
                    period=str(args.period),
                    start_ms=clamped_start_ms,
                    end_ms=end_ms,
                    limit=int(args.limit),
                    max_pages=int(args.max_pages),
                    sleep_sec=float(args.sleep_sec),
                    contract_type=str(args.contract_type),
                )
                raw_points = int(len(series))
                if bool(args.expand_to_period) and raw_points > 0:
                    series = _expand_forward_fill(
                        series,
                        start_ms=clamped_start_ms,
                        end_ms=end_ms,
                        bucket_ms=bucket_ms,
                    )
                if metric_name == "basis_rate":
                    basis_series_for_annualized = list(series)
                if metric_name == "annualized_basis_rate" and not series and basis_series_for_annualized:
                    series = [(int(ts_ms), float(val) * 365.0) for ts_ms, val in basis_series_for_annualized]
                if warn:
                    msg = f"{metric_name}:{warn}"
                    sym_obj["warnings"].append(msg)
                    out["warnings"].append(f"{sym}:{msg}")
                written = _insert_series(
                    conn,
                    symbol=sym,
                    chain=str(args.chain),
                    source=str(args.source),
                    metric_name=str(metric_name),
                    series=series,
                    batch_size=int(args.batch_size),
                )
                sym_obj["metrics"][metric_name] = {
                    "raw_points": int(raw_points),
                    "points": int(len(series)),
                    "written": int(written),
                }

            out["per_symbol"][sym] = sym_obj
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
