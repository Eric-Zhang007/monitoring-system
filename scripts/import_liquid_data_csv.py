#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")


def _parse_ts(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty_ts")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _read_csv(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(dict(row))
    return out


def _normalize_market_rows(rows: Sequence[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    out: List[Tuple[Any, ...]] = []
    seen = set()
    for r in rows:
        key = (str(r.get("symbol") or "").upper(), str(r.get("timeframe") or "").lower(), str(r.get("ts") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            (
                key[0],
                key[1],
                _parse_ts(key[2]),
                float(r.get("open") or 0.0),
                float(r.get("high") or 0.0),
                float(r.get("low") or 0.0),
                float(r.get("close") or 0.0),
                float(r.get("volume") or 0.0),
                int(float(r.get("trades_count") or 0.0)),
                str(r.get("source") or "import_csv").strip(),
            )
        )
    return out


def _normalize_orderbook_rows(rows: Sequence[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    out: List[Tuple[Any, ...]] = []
    seen = set()
    for r in rows:
        symbol = str(r.get("symbol") or "").upper()
        ts_raw = str(r.get("ts") or "")
        source = str(r.get("source") or "import_csv").strip()
        key = (symbol, ts_raw, source)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            (
                symbol,
                _parse_ts(ts_raw),
                float(r.get("bid_px") or 0.0),
                float(r.get("ask_px") or 0.0),
                float(r.get("bid_sz") or 0.0),
                float(r.get("ask_sz") or 0.0),
                float(r.get("spread_bps") or 0.0),
                float(r.get("imbalance") or 0.0),
                source,
            )
        )
    return out


def _normalize_funding_rows(rows: Sequence[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    out: List[Tuple[Any, ...]] = []
    seen = set()
    for r in rows:
        symbol = str(r.get("symbol") or "").upper()
        ts_raw = str(r.get("ts") or "")
        key = (symbol, ts_raw)
        if key in seen:
            continue
        seen.add(key)
        next_ts_raw = str(r.get("next_funding_ts") or "").strip()
        out.append(
            (
                symbol,
                _parse_ts(ts_raw),
                float(r.get("funding_rate") or 0.0),
                _parse_ts(next_ts_raw) if next_ts_raw else None,
                str(r.get("source") or "import_csv").strip(),
            )
        )
    return out


def _normalize_onchain_rows(rows: Sequence[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    out: List[Tuple[Any, ...]] = []
    seen = set()
    for r in rows:
        asset = str(r.get("asset_symbol") or "").upper()
        chain = str(r.get("chain") or "").strip()
        ts_raw = str(r.get("ts") or "")
        metric_name = str(r.get("metric_name") or "").strip()
        source = str(r.get("source") or "import_csv").strip()
        key = (asset, chain, ts_raw, metric_name, source)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            (
                asset,
                chain,
                _parse_ts(ts_raw),
                metric_name,
                float(r.get("metric_value") or 0.0),
                source,
            )
        )
    return out


def _range(rows: Sequence[Tuple[Any, ...]], ts_idx: int) -> Tuple[Optional[datetime], Optional[datetime]]:
    if not rows:
        return None, None
    values = [r[ts_idx] for r in rows if isinstance(r[ts_idx], datetime)]
    if not values:
        return None, None
    return min(values), max(values)


def _symbol_set(rows: Sequence[Tuple[Any, ...]], symbol_idx: int) -> List[str]:
    out = sorted({str(r[symbol_idx]).upper() for r in rows if str(r[symbol_idx]).strip()})
    return out


def _chunked(rows: Sequence[Tuple[Any, ...]], size: int = 1000) -> Iterable[Sequence[Tuple[Any, ...]]]:
    n = max(1, int(size))
    for i in range(0, len(rows), n):
        yield rows[i : i + n]


def _string_set(rows: Sequence[Tuple[Any, ...]], idx: int) -> List[str]:
    return sorted({str(r[idx]).strip() for r in rows if str(r[idx]).strip()})


def main() -> int:
    ap = argparse.ArgumentParser(description="Import market/orderbook/funding/onchain CSV files into PostgreSQL")
    ap.add_argument("--database-url", default=DATABASE_URL)
    ap.add_argument("--market-csv", default="")
    ap.add_argument("--orderbook-csv", default="")
    ap.add_argument("--funding-csv", default="")
    ap.add_argument("--onchain-csv", default="")
    ap.add_argument("--replace-window", action="store_true", default=True)
    ap.add_argument("--no-replace-window", dest="replace_window", action="store_false")
    ap.add_argument("--batch-size", type=int, default=1200)
    args = ap.parse_args()

    market_rows: List[Tuple[Any, ...]] = []
    orderbook_rows: List[Tuple[Any, ...]] = []
    funding_rows: List[Tuple[Any, ...]] = []
    onchain_rows: List[Tuple[Any, ...]] = []

    if str(args.market_csv).strip() and os.path.exists(str(args.market_csv)):
        market_rows = _normalize_market_rows(_read_csv(str(args.market_csv)))
    if str(args.orderbook_csv).strip() and os.path.exists(str(args.orderbook_csv)):
        orderbook_rows = _normalize_orderbook_rows(_read_csv(str(args.orderbook_csv)))
    if str(args.funding_csv).strip() and os.path.exists(str(args.funding_csv)):
        funding_rows = _normalize_funding_rows(_read_csv(str(args.funding_csv)))
    if str(args.onchain_csv).strip() and os.path.exists(str(args.onchain_csv)):
        onchain_rows = _normalize_onchain_rows(_read_csv(str(args.onchain_csv)))

    stats: Dict[str, Any] = {
        "status": "ok",
        "replace_window": bool(args.replace_window),
        "input_rows": {
            "market": len(market_rows),
            "orderbook": len(orderbook_rows),
            "funding": len(funding_rows),
            "onchain": len(onchain_rows),
        },
        "deleted_rows": {},
        "inserted_rows": {},
    }

    with psycopg2.connect(args.database_url) as conn:
        with conn.cursor() as cur:
            if market_rows and bool(args.replace_window):
                syms = _symbol_set(market_rows, 0)
                tfs = sorted({str(r[1]).lower() for r in market_rows})
                srcs = _string_set(market_rows, 9)
                start_ts, end_ts = _range(market_rows, 2)
                cur.execute(
                    """
                    DELETE FROM market_bars
                    WHERE symbol = ANY(%s)
                      AND timeframe = ANY(%s)
                      AND ts >= %s
                      AND ts <= %s
                      AND source = ANY(%s)
                    """,
                    (syms, tfs, start_ts, end_ts, srcs),
                )
                stats["deleted_rows"]["market"] = int(cur.rowcount or 0)

            if market_rows:
                inserted = 0
                for chunk in _chunked(market_rows, size=int(args.batch_size)):
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
                            trades_count = EXCLUDED.trades_count,
                            source = EXCLUDED.source
                        """,
                        chunk,
                        template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        page_size=max(200, int(args.batch_size)),
                    )
                    inserted += len(chunk)
                stats["inserted_rows"]["market"] = int(inserted)

            if orderbook_rows and bool(args.replace_window):
                syms = _symbol_set(orderbook_rows, 0)
                srcs = _string_set(orderbook_rows, 8)
                start_ts, end_ts = _range(orderbook_rows, 1)
                cur.execute(
                    """
                    DELETE FROM orderbook_l2
                    WHERE symbol = ANY(%s)
                      AND ts >= %s
                      AND ts <= %s
                      AND source = ANY(%s)
                    """,
                    (syms, start_ts, end_ts, srcs),
                )
                stats["deleted_rows"]["orderbook"] = int(cur.rowcount or 0)

            if orderbook_rows:
                inserted = 0
                for chunk in _chunked(orderbook_rows, size=int(args.batch_size)):
                    execute_values(
                        cur,
                        """
                        INSERT INTO orderbook_l2 (
                            symbol, ts, bid_px, ask_px, bid_sz, ask_sz, spread_bps, imbalance, source, created_at
                        ) VALUES %s
                        """,
                        chunk,
                        template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())",
                        page_size=max(200, int(args.batch_size)),
                    )
                    inserted += len(chunk)
                stats["inserted_rows"]["orderbook"] = int(inserted)

            if funding_rows and bool(args.replace_window):
                syms = _symbol_set(funding_rows, 0)
                srcs = _string_set(funding_rows, 4)
                start_ts, end_ts = _range(funding_rows, 1)
                cur.execute(
                    """
                    DELETE FROM funding_rates
                    WHERE symbol = ANY(%s)
                      AND ts >= %s
                      AND ts <= %s
                      AND source = ANY(%s)
                    """,
                    (syms, start_ts, end_ts, srcs),
                )
                stats["deleted_rows"]["funding"] = int(cur.rowcount or 0)

            if funding_rows:
                inserted = 0
                for chunk in _chunked(funding_rows, size=int(args.batch_size)):
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
                        chunk,
                        template="(%s,%s,%s,%s,%s,NOW())",
                        page_size=max(200, int(args.batch_size)),
                    )
                    inserted += len(chunk)
                stats["inserted_rows"]["funding"] = int(inserted)

            if onchain_rows and bool(args.replace_window):
                syms = _symbol_set(onchain_rows, 0)
                chains = _string_set(onchain_rows, 1)
                metrics = _string_set(onchain_rows, 3)
                srcs = _string_set(onchain_rows, 5)
                start_ts, end_ts = _range(onchain_rows, 2)
                cur.execute(
                    """
                    DELETE FROM onchain_signals
                    WHERE asset_symbol = ANY(%s)
                      AND chain = ANY(%s)
                      AND metric_name = ANY(%s)
                      AND ts >= %s
                      AND ts <= %s
                      AND source = ANY(%s)
                    """,
                    (syms, chains, metrics, start_ts, end_ts, srcs),
                )
                stats["deleted_rows"]["onchain"] = int(cur.rowcount or 0)

            if onchain_rows:
                inserted = 0
                for chunk in _chunked(onchain_rows, size=int(args.batch_size)):
                    execute_values(
                        cur,
                        """
                        INSERT INTO onchain_signals (
                            asset_symbol, chain, ts, metric_name, metric_value, source, created_at
                        ) VALUES %s
                        """,
                        chunk,
                        template="(%s,%s,%s,%s,%s,%s,NOW())",
                        page_size=max(200, int(args.batch_size)),
                    )
                    inserted += len(chunk)
                stats["inserted_rows"]["onchain"] = int(inserted)

    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
