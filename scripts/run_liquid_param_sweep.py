#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests


def _vals(raw: str) -> List[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _score(metric: Dict[str, Any]) -> float:
    sharpe = float(metric.get("sharpe") or 0.0)
    pnl = float(metric.get("pnl_after_cost") or 0.0)
    maxdd = float(metric.get("max_drawdown") or 0.0)
    cov = float(metric.get("model_inference_coverage") or 0.0)
    return sharpe + 200.0 * pnl - 2.0 * maxdd + 0.1 * cov


def main() -> int:
    ap = argparse.ArgumentParser(description="Incremental sweep for liquid strategy params")
    ap.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    ap.add_argument("--run-source", default="maintenance", choices=["maintenance", "prod"])
    ap.add_argument("--targets", default="BTC")
    ap.add_argument("--lookback-days", type=int, default=60)
    ap.add_argument("--train-days", type=int, default=14)
    ap.add_argument("--test-days", type=int, default=3)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=3.0)
    ap.add_argument("--request-timeout-sec", type=float, default=20.0)
    ap.add_argument("--max-trials", type=int, default=180)
    ap.add_argument("--entry-grid", default="0.01,0.012,0.015,0.02,0.025,0.03")
    ap.add_argument("--exit-grid", default="0.003,0.005,0.008,0.01,0.012")
    ap.add_argument("--base-weight-grid", default="0.06,0.08,0.1,0.12,0.15")
    ap.add_argument("--high-vol-mult-grid", default="0.3,0.4,0.5,0.55,0.65")
    ap.add_argument("--cost-lambda-grid", default="0.4,0.6,0.8,1.0,1.2,1.5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    targets = [s.strip().upper() for s in args.targets.split(",") if s.strip()]
    entries = _vals(args.entry_grid)
    exits = _vals(args.exit_grid)
    bases = _vals(args.base_weight_grid)
    hvs = _vals(args.high_vol_mult_grid)
    lambdas = _vals(args.cost_lambda_grid)

    trials: List[Dict[str, Any]] = []
    idx = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for entry, exit_z, base, hv, lam in itertools.product(entries, exits, bases, hvs, lambdas):
            if idx >= args.max_trials:
                break
            idx += 1
            payload = {
                "track": "liquid",
                "run_source": args.run_source,
                "score_source": "model",
                "require_model_artifact": True,
                "targets": targets,
                "lookback_days": int(args.lookback_days),
                "train_days": int(args.train_days),
                "test_days": int(args.test_days),
                "fee_bps": float(args.fee_bps),
                "slippage_bps": float(args.slippage_bps),
                "signal_entry_z_min": float(entry),
                "signal_exit_z_min": float(exit_z),
                "position_max_weight_base": float(base),
                "position_max_weight_high_vol_mult": float(hv),
                "cost_penalty_lambda": float(lam),
            }
            row: Dict[str, Any] = {
                "idx": idx,
                "config": {
                    "entry_z": entry,
                    "exit_z": exit_z,
                    "base_weight": base,
                    "high_vol_mult": hv,
                    "cost_lambda": lam,
                },
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            try:
                resp = requests.post(f"{args.api_base}/api/v2/backtest/run", json=payload, timeout=float(args.request_timeout_sec))
                if resp.status_code >= 300:
                    row["status"] = "http_error"
                    row["http_status"] = resp.status_code
                else:
                    body = resp.json()
                    metric = (body or {}).get("metrics") or {}
                    row["status"] = str(metric.get("status") or body.get("status") or "unknown")
                    row["run_id"] = body.get("run_id")
                    row["metric"] = metric
                    if row["status"] == "completed":
                        row["score"] = _score(metric)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = type(exc).__name__
            trials.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    ok = [t for t in trials if t.get("status") == "completed" and isinstance(t.get("metric"), dict)]
    ok.sort(key=lambda x: float(x.get("score") or -1e9), reverse=True)
    summary = {
        "trials": len(trials),
        "ok_trials": len(ok),
        "top5": [
            {
                "config": t["config"],
                "run_id": t.get("run_id"),
                "score": t.get("score"),
                "sharpe": (t.get("metric") or {}).get("sharpe"),
                "pnl_after_cost": (t.get("metric") or {}).get("pnl_after_cost"),
                "max_drawdown": (t.get("metric") or {}).get("max_drawdown"),
            }
            for t in ok[:5]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

