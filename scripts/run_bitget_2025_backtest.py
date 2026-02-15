#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from _metrics_test_logger import record_metrics_test


BITGET_PERP_URL = "https://api.bitget.com/api/v2/mix/market/history-candles"
BITGET_SPOT_URL = "https://api.bitget.com/api/v2/spot/market/history-candles"
SYMBOL_MAP = {
    "BTCUSDT": "BTC_BG2025",
    "ETHUSDT": "ETH_BG2025",
    "SOLUSDT": "SOL_BG2025",
}


@dataclass
class Candle:
    ts_ms: int
    close: float
    volume: float


@dataclass
class EventPoint:
    ts_ms: int
    tier_weight: float
    confidence: float


def _run(cmd: List[str], input_text: str | None = None) -> str:
    p = subprocess.run(cmd, input=input_text, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def _psql(sql: str) -> str:
    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "monitor",
        "-d",
        "monitor",
        "-At",
        "-F",
        "|",
        "-c",
        sql,
    ]
    return _run(cmd)


def _copy_csv(table_cols: str, csv_text: str) -> None:
    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "monitor",
        "-d",
        "monitor",
        "-c",
        f"\\copy {table_cols} FROM STDIN WITH (FORMAT csv)",
    ]
    _run(cmd, input_text=csv_text)


def _fetch_symbol_2025(symbol: str, market: str, limit: int = 200) -> List[Candle]:
    start = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000) - 1
    cursor = start
    out: List[Candle] = []
    seen = set()
    step_ms = 60 * 60 * 1000

    while cursor <= end:
        req_end = min(end, cursor + limit * step_ms - 1)
        if market == "spot":
            params = {
                "symbol": symbol,
                "granularity": "1h",
                "startTime": str(cursor),
                "endTime": str(req_end),
                "limit": str(limit),
            }
            r = requests.get(BITGET_SPOT_URL, params=params, timeout=20)
        else:
            params = {
                "symbol": symbol,
                "productType": "USDT-FUTURES",
                "granularity": "1H",
                "startTime": str(cursor),
                "endTime": str(req_end),
                "limit": str(limit),
            }
            r = requests.get(BITGET_PERP_URL, params=params, timeout=20)
        r.raise_for_status()
        body = r.json()
        if str(body.get("code")) != "00000":
            raise RuntimeError(f"bitget {market} error: {body}")
        data = body.get("data") or []
        if not data:
            break
        batch: List[Candle] = []
        for row in data:
            try:
                ts_ms = int(row[0])
                close = float(row[4])
                volume = float(row[5])
            except Exception:
                continue
            if ts_ms < start or ts_ms > end:
                continue
            if ts_ms in seen:
                continue
            seen.add(ts_ms)
            batch.append(Candle(ts_ms=ts_ms, close=close, volume=volume))
        if not batch:
            break
        batch.sort(key=lambda x: x.ts_ms)
        out.extend(batch)
        cursor = batch[-1].ts_ms + step_ms
    out.sort(key=lambda x: x.ts_ms)
    return out


def _load_event_points_2025(local_symbol_base: str, start_iso: str, end_iso: str) -> List[EventPoint]:
    sql_new = f"""
    SELECT
      (EXTRACT(EPOCH FROM COALESCE(e.available_at, e.occurred_at)) * 1000)::bigint AS ts_ms,
      LEAST(1.0, GREATEST(0.1, (6.0 - COALESCE(e.source_tier, 3)::double precision) / 5.0)) AS tier_weight,
      COALESCE(e.confidence_score, 0.5)::double precision AS confidence
    FROM events e
    LEFT JOIN event_links el ON el.event_id = e.id
    LEFT JOIN entities en ON en.id = el.entity_id
    WHERE (
      UPPER(en.symbol) = UPPER('{local_symbol_base}')
      OR en.name = '{local_symbol_base}'
      OR e.market_scope = 'macro'
      OR COALESCE(e.payload->>'global_impact', 'false') = 'true'
    )
      AND COALESCE(e.available_at, e.occurred_at) >= '{start_iso}'::timestamptz
      AND COALESCE(e.available_at, e.occurred_at) <= '{end_iso}'::timestamptz
    ORDER BY ts_ms ASC;
    """
    sql_old = f"""
    SELECT
      (EXTRACT(EPOCH FROM e.occurred_at) * 1000)::bigint AS ts_ms,
      LEAST(1.0, GREATEST(0.1, (6.0 - COALESCE(e.source_tier, 3)::double precision) / 5.0)) AS tier_weight,
      COALESCE(e.confidence_score, 0.5)::double precision AS confidence
    FROM events e
    LEFT JOIN event_links el ON el.event_id = e.id
    LEFT JOIN entities en ON en.id = el.entity_id
    WHERE (
      UPPER(en.symbol) = UPPER('{local_symbol_base}')
      OR en.name = '{local_symbol_base}'
    )
      AND e.occurred_at >= '{start_iso}'::timestamptz
      AND e.occurred_at <= '{end_iso}'::timestamptz
    ORDER BY ts_ms ASC;
    """
    raw = ""
    try:
        raw = _psql(sql_new)
    except Exception:
        raw = _psql(sql_old)
    points: List[EventPoint] = []
    for line in str(raw or "").splitlines():
        cols = line.split("|")
        if len(cols) < 3:
            continue
        try:
            ts_ms = int(float(cols[0]))
            tier_weight = float(cols[1])
            confidence = float(cols[2])
        except Exception:
            continue
        points.append(EventPoint(ts_ms=ts_ms, tier_weight=max(0.1, min(1.0, tier_weight)), confidence=max(0.0, min(1.0, confidence))))
    points.sort(key=lambda x: x.ts_ms)
    return points


def _build_feature_rows(candles: List[Candle], event_points: Optional[List[EventPoint]] = None) -> List[Tuple[int, Dict[str, float]]]:
    if len(candles) < 120:
        return []
    rows: List[Tuple[int, Dict[str, float]]] = []
    prices = [c.close for c in candles]
    vols = [max(c.volume, 0.0) for c in candles]
    event_points = event_points or []
    ev_ts = [e.ts_ms for e in event_points]
    n = len(candles)
    for i in range(96, n - 1):
        price = prices[i]
        prev_1 = prices[i - 1]
        prev_3 = prices[i - 3]
        prev_12 = prices[i - 12]
        prev_48 = prices[i - 48]
        ret_1 = (price - prev_1) / max(prev_1, 1e-12)
        ret_3 = (price - prev_3) / max(prev_3, 1e-12)
        ret_12 = (price - prev_12) / max(prev_12, 1e-12)
        ret_48 = (price - prev_48) / max(prev_48, 1e-12)

        def _std_logret(a: int, b: int) -> float:
            segment = prices[a:b]
            lrs = []
            for j in range(1, len(segment)):
                p0 = max(segment[j - 1], 1e-12)
                p1 = max(segment[j], 1e-12)
                lrs.append(math.log(p1 / p0))
            if not lrs:
                return 0.0
            m = sum(lrs) / len(lrs)
            v = sum((x - m) ** 2 for x in lrs) / len(lrs)
            return math.sqrt(max(v, 0.0))

        vol_3 = _std_logret(i - 3, i)
        vol_12 = _std_logret(i - 12, i)
        vol_48 = _std_logret(i - 48, i)
        vol_96 = _std_logret(i - 96, i)
        vol = vols[i]
        hist = vols[i - 12 : i]
        hist_mean = sum(hist) / len(hist)
        hist_var = sum((x - hist_mean) ** 2 for x in hist) / len(hist)
        hist_std = math.sqrt(max(hist_var, 1e-12))
        vol_z = (vol - hist_mean) / max(hist_std, 1e-6)
        volume_impact = abs(ret_1) / max(math.sqrt(max(vol, 1.0)), 1e-6)
        event_decay = 0.0
        source_tier_weight = 0.0
        source_confidence = 0.0
        if ev_ts:
            ts_ms = candles[i].ts_ms
            right = bisect_right(ev_ts, ts_ms) - 1
            if right >= 0:
                left = bisect_right(ev_ts, ts_ms - 48 * 3600 * 1000)
                num_decay = 0.0
                den = 0.0
                tier_sum = 0.0
                conf_sum = 0.0
                cnt = 0
                for j in range(left, right + 1):
                    ev = event_points[j]
                    age_h = max(0.0, (ts_ms - ev.ts_ms) / 3600000.0)
                    decay = math.exp(-age_h / 12.0)
                    w = max(1e-9, ev.tier_weight * ev.confidence)
                    num_decay += w * decay
                    den += w
                    tier_sum += ev.tier_weight
                    conf_sum += ev.confidence
                    cnt += 1
                if den > 0:
                    event_decay = float(num_decay / den)
                if cnt > 0:
                    source_tier_weight = float(tier_sum / cnt)
                    source_confidence = float(conf_sum / cnt)
        feat = {
            "ret_1": ret_1,
            "ret_3": ret_3,
            "ret_12": ret_12,
            "ret_48": ret_48,
            "vol_3": vol_3,
            "vol_12": vol_12,
            "vol_48": vol_48,
            "vol_96": vol_96,
            "log_volume": math.log1p(max(vol, 0.0)),
            "vol_z": vol_z,
            "volume_impact": volume_impact,
            "orderbook_imbalance": 0.0,
            "funding_rate": 0.0,
            "onchain_norm": 0.0,
            "event_decay": event_decay,
            "source_tier_weight": source_tier_weight,
            "source_confidence": source_confidence,
            "orderbook_missing_flag": 1.0,
            "funding_missing_flag": 1.0,
            "onchain_missing_flag": 1.0,
            "feature_payload_schema_version": "v2.1",
        }
        rows.append((candles[i].ts_ms, feat))
    return rows


def _to_csv_text(rows: List[List[str]]) -> str:
    with tempfile.NamedTemporaryFile("w+", newline="", encoding="utf-8", delete=False) as tmp:
        writer = csv.writer(tmp)
        writer.writerows(rows)
        path = tmp.name
    text = Path(path).read_text(encoding="utf-8")
    Path(path).unlink(missing_ok=True)
    return text


def _run_single_market(api_base: str, run_source: str, lookback_days: int, market: str) -> Dict[str, object]:
    started = datetime.now(timezone.utc)
    start_2025 = "2025-01-01T00:00:00+00:00"
    end_2025 = "2025-12-31T23:59:59+00:00"
    ingest_stats: Dict[str, Dict[str, int]] = {}
    suffix = "SPOT" if market == "spot" else "PERP"

    for remote_sym, local_sym_base in SYMBOL_MAP.items():
        local_sym = f"{local_sym_base}_{suffix}"
        candles = _fetch_symbol_2025(remote_sym, market=market)
        if len(candles) < 200:
            raise RuntimeError(f"insufficient candles for {remote_sym} ({market}): {len(candles)}")
        event_points = _load_event_points_2025(local_sym_base, start_2025, end_2025)

        _psql(
            "DELETE FROM prices "
            f"WHERE symbol='{local_sym}' "
            f"AND timestamp >= '{start_2025}'::timestamptz "
            f"AND timestamp <= '{end_2025}'::timestamptz;"
        )
        _psql(
            "DELETE FROM feature_snapshots "
            f"WHERE target='{local_sym}' AND track='liquid' "
            f"AND as_of_ts >= '{start_2025}'::timestamptz "
            f"AND as_of_ts <= '{end_2025}'::timestamptz;"
        )

        price_rows = []
        for c in candles:
            ts = datetime.fromtimestamp(c.ts_ms / 1000.0, tz=timezone.utc).isoformat()
            price_rows.append([local_sym, f"{c.close:.8f}", str(int(c.volume)), ts])
        _copy_csv("prices(symbol,price,volume,timestamp)", _to_csv_text(price_rows))

        feature_rows = _build_feature_rows(candles, event_points=event_points)
        feature_csv_rows = []
        lineage_id = f"bitget-2025-{market}-{local_sym.lower()}-1h"
        data_version = f"bitget_2025_{market}_1h"
        for ts_ms, feat in feature_rows:
            ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()
            feature_csv_rows.append(
                [
                    local_sym,
                    "liquid",
                    ts,
                    "feature-store-v2.0",
                    json.dumps(feat, ensure_ascii=False),
                    ts,
                    ts,
                    ts,
                    data_version,
                    lineage_id,
                    ts,
                ]
            )
        csv_text = _to_csv_text(feature_csv_rows)
        try:
            _copy_csv(
                "feature_snapshots(target,track,as_of,feature_version,feature_payload,as_of_ts,event_time,feature_available_at,data_version,lineage_id,created_at)",
                csv_text,
            )
        except Exception:
            stripped = [row[:7] + row[8:] for row in feature_csv_rows]
            _copy_csv(
                "feature_snapshots(target,track,as_of,feature_version,feature_payload,as_of_ts,event_time,data_version,lineage_id,created_at)",
                _to_csv_text(stripped),
            )
        ingest_stats[local_sym] = {"prices": len(price_rows), "features": len(feature_csv_rows)}

    payload = {
        "track": "liquid",
        "run_source": run_source,
        "targets": [f"{s}_{suffix}" for s in SYMBOL_MAP.values()],
        "horizon": "1d",
        "data_version": f"bitget_2025_{market}_1h",
        "lookback_days": int(lookback_days),
        "train_days": 35,
        "test_days": 7,
        "fee_bps": 5.0,
        "slippage_bps": 3.0,
        "alignment_mode": "strict_asof",
        "alignment_version": "strict_asof_v1",
        "max_feature_staleness_hours": 24 * 14,
        "signal_polarity_mode": "auto_train_ic",
    }
    r = requests.post(f"{api_base}/api/v2/backtest/run", json=payload, timeout=120)
    r.raise_for_status()
    bt = r.json()
    metrics = (bt or {}).get("metrics") or {}
    out = {
        "test_name": f"bitget_2025_real_history_backtest_{market}",
        "executed_at": started.isoformat(),
        "window_start": start_2025,
        "window_end": end_2025,
        "source": {
            "exchange": "bitget",
            "market": "spot" if market == "spot" else "USDT-FUTURES",
            "granularity": "1h" if market == "spot" else "1H",
            "symbols": SYMBOL_MAP,
        },
        "ingest_stats": ingest_stats,
        "backtest_request": payload,
        "backtest_response": {
            "run_id": bt.get("run_id"),
            "status": bt.get("status"),
            "metrics": metrics,
        },
        "key_metrics": {
            "samples": metrics.get("samples"),
            "ic": metrics.get("ic"),
            "sharpe": metrics.get("sharpe"),
            "hit_rate": metrics.get("hit_rate"),
            "turnover": metrics.get("turnover"),
            "pnl_after_cost": metrics.get("pnl_after_cost"),
            "max_drawdown": metrics.get("max_drawdown"),
            "lineage_coverage": metrics.get("lineage_coverage"),
            "cost_breakdown": metrics.get("cost_breakdown"),
            "regime_breakdown": metrics.get("regime_breakdown"),
        },
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Bitget 2025 real-history backtest runner")
    ap.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    ap.add_argument("--run-source", default="maintenance", choices=["prod", "maintenance"])
    ap.add_argument("--lookback-days", type=int, default=500)
    ap.add_argument("--market", default="perp", choices=["perp", "spot", "both"])
    args = ap.parse_args()

    markets = ["perp", "spot"] if args.market == "both" else [args.market]
    runs: Dict[str, Dict[str, object]] = {}
    for m in markets:
        runs[m] = _run_single_market(
            api_base=args.api_base,
            run_source=args.run_source,
            lookback_days=args.lookback_days,
            market=m,
        )
    out = {
        "test_name": "bitget_2025_real_history_backtest",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "markets": runs,
    }

    out_dir = Path("artifacts/bitget_2025")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))
    print(f"saved_report={out_path}")

    record_metrics_test(
        test_name="bitget_2025_real_history_backtest",
        payload=out,
        window_start="2025-01-01T00:00:00+00:00",
        window_end="2025-12-31T23:59:59+00:00",
        extra={"report_path": str(out_path)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
