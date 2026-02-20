#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import requests
from _psql import run_psql


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay one backtest run configuration")
    ap.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    ap.add_argument("--run-id", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    args = ap.parse_args()

    sql = (
        f"SELECT json_build_object('id',id,'track',track,'config',config,'metrics',metrics)::text FROM backtest_runs WHERE id = {args.run_id} LIMIT 1;"
        if args.run_id > 0
        else "SELECT json_build_object('id',id,'track',track,'config',config,'metrics',metrics)::text FROM backtest_runs ORDER BY id DESC LIMIT 1;"
    )
    raw = run_psql(sql)
    if not raw:
        print(json.dumps({"status": "failed", "reason": "run_not_found"}, ensure_ascii=False))
        return 2
    row = json.loads(raw)

    cfg = dict(row.get("config") or {})
    payload = {
        "track": cfg.get("track") or row.get("track") or "liquid",
        "targets": cfg.get("targets") or [],
        "horizon": cfg.get("horizon") or "1d",
        "model_name": cfg.get("model_name"),
        "model_version": cfg.get("model_version"),
        "data_version": cfg.get("data_version") or "v1",
        "lookback_days": int(cfg.get("lookback_days") or 90),
        "train_days": int(cfg.get("train_days") or 35),
        "test_days": int(cfg.get("test_days") or 7),
        "fee_bps": float(cfg.get("fee_bps") or 5.0),
        "slippage_bps": float(cfg.get("slippage_bps") or 3.0),
    }

    replay = requests.post(f"{args.api_base}/api/v2/backtest/run", json=payload, timeout=60)
    replay.raise_for_status()
    replay_data = replay.json()

    old_metrics = row.get("metrics") or {}
    new_metrics = replay_data.get("metrics") or {}
    keys = ["ic", "hit_rate", "turnover", "pnl_after_cost", "max_drawdown", "lineage_coverage"]
    diffs = {k: abs(float(new_metrics.get(k, 0.0)) - float(old_metrics.get(k, 0.0))) for k in keys}
    passed = all(v <= args.tolerance for v in diffs.values())

    out = {
        "source_run_id": int(row.get("id")),
        "replay_run_id": int(replay_data.get("run_id")),
        "tolerance": args.tolerance,
        "diffs": {k: round(v, 12) for k, v in diffs.items()},
        "passed": passed,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
