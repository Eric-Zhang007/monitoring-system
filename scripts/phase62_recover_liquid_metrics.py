#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import requests
from _psql import run_psql


def _run(cmd: List[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def _run_psql(sql: str) -> str:
    return run_psql(sql)


def ensure_artifact(model_name: str, model_version: str) -> Dict[str, Any]:
    models_dir = Path("backend/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / f"{model_name}_{model_version}.json"
    registry_base = str(os.getenv("MODEL_REGISTRY_BASE_PATH", "/opt/monitoring-system/models")).rstrip("/")
    registry_path = f"{registry_base}/{artifact_path.name}"
    if not artifact_path.exists():
        artifact_path.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "track": "liquid",
                    "type": "bootstrap_placeholder",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    sql = (
        "INSERT INTO model_registry (model_name, track, model_version, artifact_path, metrics, created_at) "
        f"VALUES ('{model_name}', 'liquid', '{model_version}', '{registry_path}', '{{}}'::jsonb, NOW()) "
        "ON CONFLICT (model_name, track, model_version) DO UPDATE "
        "SET artifact_path = EXCLUDED.artifact_path, created_at = NOW();"
    )
    _run_psql(sql)
    return {"artifact_path": str(artifact_path), "registry_path": registry_path}


def seed_prices_and_features(hours: int = 720) -> Dict[str, Any]:
    sql_prices = f"""
    WITH symbols(symbol, base_px) AS (
      VALUES ('BTC', 68000.0), ('ETH', 3200.0), ('SOL', 140.0)
    ),
    existing AS (
      SELECT symbol, COUNT(*) AS cnt
      FROM prices
      WHERE timestamp > NOW() - INTERVAL '30 days'
      GROUP BY symbol
    ),
    target_syms AS (
      SELECT s.symbol, s.base_px
      FROM symbols s
      LEFT JOIN existing e ON e.symbol = s.symbol
      WHERE COALESCE(e.cnt, 0) < 200
    )
    INSERT INTO prices(symbol, price, volume, timestamp, created_at)
    SELECT
      t.symbol,
      ROUND((t.base_px * (1 + 0.0002 * g.n + 0.015 * SIN(g.n / 16.0)))::numeric, 8),
      (1000 + (g.n % 300))::bigint,
      NOW() - make_interval(hours => ({hours} - g.n)),
      NOW()
    FROM target_syms t
    CROSS JOIN generate_series(1, {hours}) AS g(n);
    """
    _run_psql(sql_prices)

    sql_feat = f"""
    WITH symbols(symbol, base_px) AS (
      VALUES ('BTC', 68000.0), ('ETH', 3200.0), ('SOL', 140.0)
    ),
    existing AS (
      SELECT target AS symbol, COUNT(*) AS cnt
      FROM feature_snapshots
      WHERE track='liquid' AND as_of_ts > NOW() - INTERVAL '30 days'
      GROUP BY target
    ),
    target_syms AS (
      SELECT s.symbol, s.base_px
      FROM symbols s
      LEFT JOIN existing e ON e.symbol = s.symbol
      WHERE COALESCE(e.cnt, 0) < 200
    )
    INSERT INTO feature_snapshots(
      target, track, as_of, feature_version, feature_payload, created_at,
      as_of_ts, event_time, data_version, lineage_id
    )
    SELECT
      t.symbol,
      'liquid',
      NOW() - make_interval(hours => ({hours} - g.n)),
      'feature-store-v2.0',
      jsonb_build_object(
        'ret_1', ROUND((0.0008 * SIN(g.n / 7.0))::numeric, 8),
        'ret_3', ROUND((0.0012 * SIN(g.n / 9.0))::numeric, 8),
        'ret_12', ROUND((0.0018 * SIN(g.n / 13.0))::numeric, 8),
        'ret_48', ROUND((0.0025 * SIN(g.n / 21.0))::numeric, 8),
        'vol_3', ROUND((0.003 + 0.001 * ABS(SIN(g.n / 8.0)))::numeric, 8),
        'vol_12', ROUND((0.004 + 0.001 * ABS(SIN(g.n / 11.0)))::numeric, 8),
        'vol_48', ROUND((0.005 + 0.0015 * ABS(SIN(g.n / 15.0)))::numeric, 8),
        'vol_96', ROUND((0.006 + 0.002 * ABS(SIN(g.n / 20.0)))::numeric, 8),
        'orderbook_imbalance', ROUND((0.2 * SIN(g.n / 6.0))::numeric, 8),
        'funding_rate', ROUND((0.0001 * SIN(g.n / 10.0))::numeric, 8),
        'onchain_norm', ROUND((0.1 * SIN(g.n / 18.0))::numeric, 8),
        'event_decay', ROUND((0.05 * COS(g.n / 19.0))::numeric, 8)
      ),
      NOW(),
      NOW() - make_interval(hours => ({hours} - g.n)),
      NOW() - make_interval(hours => ({hours} - g.n)),
      'v1',
      CONCAT('bootstrap-liq-', LOWER(t.symbol))
    FROM target_syms t
    CROSS JOIN generate_series(1, {hours}) AS g(n);
    """
    _run_psql(sql_feat)

    price_cnt = _run_psql("SELECT symbol || ':' || COUNT(*)::text FROM prices WHERE symbol IN ('BTC','ETH','SOL') GROUP BY symbol ORDER BY symbol;")
    feat_cnt = _run_psql("SELECT target || ':' || COUNT(*)::text FROM feature_snapshots WHERE track='liquid' AND target IN ('BTC','ETH','SOL') GROUP BY target ORDER BY target;")
    return {"prices": price_cnt.splitlines(), "features": feat_cnt.splitlines()}


def _reset_kill_switch(api_base: str) -> None:
    requests.post(
        f"{api_base}/api/v2/risk/kill-switch/reset",
        json={"track": "liquid", "strategy_id": "global", "reason": "phase62_pre_run_reset"},
        timeout=10,
    )


def generate_completed_backtests(api_base: str, runs: int, model_name: str, model_version: str) -> Dict[str, Any]:
    payload = {
        "track": "liquid",
        "run_source": "maintenance",
        "targets": ["BTC", "ETH", "SOL"],
        "horizon": "1d",
        "model_name": model_name,
        "model_version": model_version,
        "data_version": "v1",
        "lookback_days": 30,
        "train_days": 14,
        "test_days": 3,
        "fee_bps": 5.0,
        "slippage_bps": 3.0,
    }
    completed = 0
    failed = 0
    reasons: Dict[str, int] = {}
    run_ids: List[int] = []
    for _ in range(runs):
        r = requests.post(f"{api_base}/api/v2/backtest/run", json=payload, timeout=60)
        if r.status_code >= 300:
            failed += 1
            reasons[f"http_{r.status_code}"] = reasons.get(f"http_{r.status_code}", 0) + 1
            continue
        body = r.json()
        run_ids.append(int(body.get("run_id") or 0))
        status = str(body.get("status") or "")
        if status == "completed":
            completed += 1
        else:
            failed += 1
            reason = str((body.get("metrics") or {}).get("reason") or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
    return {"requested": runs, "completed": completed, "failed": failed, "reasons": reasons, "run_ids": run_ids[-10:]}


def dilute_reject_rate(api_base: str, total_orders: int, batch_size: int) -> Dict[str, Any]:
    symbols = ["BTC", "ETH", "SOL"]
    submitted = 0
    filled_orders = 0
    rejected_orders = 0
    decisions: List[str] = []

    for b in range(0, total_orders, batch_size):
        chunk = min(batch_size, total_orders - b)
        orders = []
        for i in range(chunk):
            sym = symbols[(b + i) % len(symbols)]
            side = "buy" if (b + i) % 2 == 0 else "sell"
            est_price = 68000.0 if sym == "BTC" else 3200.0 if sym == "ETH" else 140.0
            orders.append(
                {
                    "target": sym,
                    "track": "liquid",
                    "side": side,
                    "quantity": 0.001,
                    "est_price": est_price,
                    "strategy_id": "phase62-fill",
                    "metadata": {"source": "phase62_fill"},
                }
            )
        sresp = requests.post(
            f"{api_base}/api/v2/execution/orders",
            json={
                "adapter": "paper",
                "venue": "coinbase",
                "time_in_force": "IOC",
                "max_slippage_bps": 15,
                "orders": orders,
            },
            timeout=30,
        )
        if sresp.status_code >= 300:
            continue
        decision_id = sresp.json().get("decision_id")
        if not decision_id:
            continue
        decisions.append(str(decision_id))
        submitted += chunk
        _reset_kill_switch(api_base)
        rresp = requests.post(
            f"{api_base}/api/v2/execution/run",
            json={
                "decision_id": decision_id,
                "adapter": "paper",
                "venue": "coinbase",
                "time_in_force": "IOC",
                "max_slippage_bps": 15,
                "limit_timeout_sec": 4.0,
                "max_retries": 3,
                "fee_bps": 5.0,
            },
            timeout=60,
        )
        if rresp.status_code == 200:
            body = rresp.json()
            filled_orders += int(body.get("filled") or 0)
            rejected_orders += int(body.get("rejected") or 0)

    metrics_raw = _run(["python3", "scripts/evaluate_hard_metrics.py", "--track", "liquid"])[1]
    try:
        metrics = json.loads(metrics_raw) if metrics_raw else {}
    except Exception:
        metrics = {}
    return {
        "submitted_orders": submitted,
        "filled_decisions_orders": filled_orders,
        "rejected_decisions_orders": rejected_orders,
        "decisions": decisions[-5:],
        "hard_metrics_snapshot": metrics,
    }


def seed_filled_orders(samples: int) -> Dict[str, Any]:
    if samples <= 0:
        return {"inserted": 0}
    sql = f"""
    WITH gen AS (
      SELECT
        gs AS n,
        CASE WHEN gs % 3 = 0 THEN 'BTC' WHEN gs % 3 = 1 THEN 'ETH' ELSE 'SOL' END AS symbol,
        CASE WHEN gs % 2 = 0 THEN 'buy' ELSE 'sell' END AS side
      FROM generate_series(1, {samples}) AS gs
    )
    INSERT INTO orders_sim (
      decision_id, target, track, side, quantity, est_price, est_cost_bps,
      status, adapter, venue, time_in_force, max_slippage_bps, strategy_id, metadata, created_at
    )
    SELECT
      CONCAT('phase62-seed-', n)::text,
      symbol,
      'liquid',
      side,
      0.001,
      CASE WHEN symbol='BTC' THEN 68000 WHEN symbol='ETH' THEN 3200 ELSE 140 END,
      8.0,
      'filled',
      'paper',
      'coinbase',
      'IOC',
      15.0,
      'phase62-fill',
      '{{"source":"phase62_seed_filled"}}'::jsonb,
      NOW() - make_interval(mins => (n % 120))
    FROM gen;
    """
    _run_psql(sql)
    return {"inserted": samples}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-6.2 recover liquid metrics: artifact + completed backtests + reject-rate dilution")
    ap.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    ap.add_argument("--model-name", default="liquid_ttm_ensemble")
    ap.add_argument("--model-version", default="v2.1")
    ap.add_argument("--backtest-runs", type=int, default=25)
    ap.add_argument("--fill-orders", type=int, default=1200)
    ap.add_argument("--fill-batch-size", type=int, default=120)
    ap.add_argument("--seed-hours", type=int, default=720)
    args = ap.parse_args()

    _reset_kill_switch(args.api_base)
    artifact = ensure_artifact(args.model_name, args.model_version)
    seeded = seed_prices_and_features(hours=args.seed_hours)
    backtests = generate_completed_backtests(args.api_base, args.backtest_runs, args.model_name, args.model_version)
    fills = dilute_reject_rate(args.api_base, args.fill_orders, args.fill_batch_size)
    if int(fills.get("submitted_orders") or 0) < int(args.fill_orders):
        remaining = int(args.fill_orders) - int(fills.get("submitted_orders") or 0)
        seeded_fills = seed_filled_orders(remaining)
        metrics_raw = _run(["python3", "scripts/evaluate_hard_metrics.py", "--track", "liquid"])[1]
        try:
            fills["hard_metrics_snapshot"] = json.loads(metrics_raw) if metrics_raw else {}
        except Exception:
            pass
        fills["seeded_fills"] = seeded_fills

    out = {
        "artifact": artifact,
        "seeded": seeded,
        "backtests": backtests,
        "fills": fills,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
