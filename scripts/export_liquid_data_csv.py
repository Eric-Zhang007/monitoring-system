#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    cnt = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})
            cnt += 1
    return int(cnt)


def _dt_field(row: Dict[str, Any], key: str) -> str:
    val = row.get(key)
    if isinstance(val, datetime):
        return _to_iso_z(val)
    return str(val or "")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export liquid market/aux tables to CSV bundle")
    ap.add_argument("--database-url", default=DATABASE_URL)
    ap.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK"))
    ap.add_argument("--timeframe", default=os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m"))
    ap.add_argument("--start", default="2018-01-01T00:00:00Z")
    ap.add_argument("--end", default="")
    ap.add_argument("--out-dir", default="artifacts/offline_bundle/latest")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise RuntimeError("empty_symbols")

    start_dt = _parse_dt_utc(args.start)
    end_dt = _parse_dt_utc(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    if end_dt <= start_dt:
        raise RuntimeError("invalid_time_range")

    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = str(args.prefix).strip()
    if pfx:
        pfx = pfx + "_"

    market_file = out_dir / f"{pfx}market_bars_{str(args.timeframe).lower()}.csv"
    orderbook_file = out_dir / f"{pfx}orderbook_l2.csv"
    funding_file = out_dir / f"{pfx}funding_rates.csv"
    onchain_file = out_dir / f"{pfx}onchain_signals.csv"

    manifest: Dict[str, Any] = {
        "status": "ok",
        "generated_at": _to_iso_z(datetime.now(timezone.utc)),
        "window": {
            "start": _to_iso_z(start_dt),
            "end": _to_iso_z(end_dt),
            "timeframe": str(args.timeframe).lower(),
        },
        "symbols": symbols,
        "files": {},
    }

    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT symbol, timeframe, ts, open, high, low, close, volume, trades_count, source
                FROM market_bars
                WHERE timeframe = %s
                  AND symbol = ANY(%s)
                  AND ts >= %s
                  AND ts <= %s
                ORDER BY symbol, ts
                """,
                (str(args.timeframe).lower(), symbols, start_dt, end_dt),
            )
            market_rows_raw = [dict(r) for r in (cur.fetchall() or [])]

            cur.execute(
                """
                SELECT symbol, ts, bid_px, ask_px, bid_sz, ask_sz, spread_bps, imbalance, source
                FROM orderbook_l2
                WHERE symbol = ANY(%s)
                  AND ts >= %s
                  AND ts <= %s
                ORDER BY symbol, ts
                """,
                (symbols, start_dt, end_dt),
            )
            orderbook_rows_raw = [dict(r) for r in (cur.fetchall() or [])]

            cur.execute(
                """
                SELECT symbol, ts, funding_rate, next_funding_ts, source
                FROM funding_rates
                WHERE symbol = ANY(%s)
                  AND ts >= %s
                  AND ts <= %s
                ORDER BY symbol, ts
                """,
                (symbols, start_dt, end_dt),
            )
            funding_rows_raw = [dict(r) for r in (cur.fetchall() or [])]

            cur.execute(
                """
                SELECT asset_symbol, chain, ts, metric_name, metric_value, source
                FROM onchain_signals
                WHERE asset_symbol = ANY(%s)
                  AND ts >= %s
                  AND ts <= %s
                ORDER BY asset_symbol, ts
                """,
                (symbols, start_dt, end_dt),
            )
            onchain_rows_raw = [dict(r) for r in (cur.fetchall() or [])]

    market_rows = [
        {
            "symbol": str(r.get("symbol") or ""),
            "timeframe": str(r.get("timeframe") or ""),
            "ts": _dt_field(r, "ts"),
            "open": float(r.get("open") or 0.0),
            "high": float(r.get("high") or 0.0),
            "low": float(r.get("low") or 0.0),
            "close": float(r.get("close") or 0.0),
            "volume": float(r.get("volume") or 0.0),
            "trades_count": int(r.get("trades_count") or 0),
            "source": str(r.get("source") or "unknown"),
        }
        for r in market_rows_raw
    ]
    orderbook_rows = [
        {
            "symbol": str(r.get("symbol") or ""),
            "ts": _dt_field(r, "ts"),
            "bid_px": float(r.get("bid_px") or 0.0),
            "ask_px": float(r.get("ask_px") or 0.0),
            "bid_sz": float(r.get("bid_sz") or 0.0),
            "ask_sz": float(r.get("ask_sz") or 0.0),
            "spread_bps": float(r.get("spread_bps") or 0.0),
            "imbalance": float(r.get("imbalance") or 0.0),
            "source": str(r.get("source") or "unknown"),
        }
        for r in orderbook_rows_raw
    ]
    funding_rows = [
        {
            "symbol": str(r.get("symbol") or ""),
            "ts": _dt_field(r, "ts"),
            "funding_rate": float(r.get("funding_rate") or 0.0),
            "next_funding_ts": _dt_field(r, "next_funding_ts") if r.get("next_funding_ts") else "",
            "source": str(r.get("source") or "unknown"),
        }
        for r in funding_rows_raw
    ]
    onchain_rows = [
        {
            "asset_symbol": str(r.get("asset_symbol") or ""),
            "chain": str(r.get("chain") or ""),
            "ts": _dt_field(r, "ts"),
            "metric_name": str(r.get("metric_name") or ""),
            "metric_value": float(r.get("metric_value") or 0.0),
            "source": str(r.get("source") or "unknown"),
        }
        for r in onchain_rows_raw
    ]

    market_n = _write_csv(
        market_file,
        ["symbol", "timeframe", "ts", "open", "high", "low", "close", "volume", "trades_count", "source"],
        market_rows,
    )
    orderbook_n = _write_csv(
        orderbook_file,
        ["symbol", "ts", "bid_px", "ask_px", "bid_sz", "ask_sz", "spread_bps", "imbalance", "source"],
        orderbook_rows,
    )
    funding_n = _write_csv(
        funding_file,
        ["symbol", "ts", "funding_rate", "next_funding_ts", "source"],
        funding_rows,
    )
    onchain_n = _write_csv(
        onchain_file,
        ["asset_symbol", "chain", "ts", "metric_name", "metric_value", "source"],
        onchain_rows,
    )

    manifest["files"] = {
        "market_bars_csv": {"path": str(market_file), "rows": int(market_n)},
        "orderbook_l2_csv": {"path": str(orderbook_file), "rows": int(orderbook_n)},
        "funding_rates_csv": {"path": str(funding_file), "rows": int(funding_n)},
        "onchain_signals_csv": {"path": str(onchain_file), "rows": int(onchain_n)},
    }

    manifest_path = out_dir / f"{pfx}manifest_liquid_data.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps({"status": "ok", "manifest": str(manifest_path), "files": manifest["files"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
