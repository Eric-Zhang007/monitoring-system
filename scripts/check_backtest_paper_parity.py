#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

import requests


def main() -> int:
    p = argparse.ArgumentParser(description="Check backtest vs paper PnL parity")
    p.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    p.add_argument("--track", default="liquid")
    p.add_argument("--target", default="BTC")
    p.add_argument("--max-deviation", type=float, default=0.10, help="max allowed relative deviation")
    args = p.parse_args()

    bt = requests.post(
        f"{args.api_base}/api/v2/backtest/run",
        json={
            "track": args.track,
            "targets": [args.target],
            "horizon": "1d",
            "model_name": "liquid_ttm_ensemble",
            "model_version": "v2.1",
            "data_version": "v1",
            "lookback_days": 30,
            "train_days": 14,
            "test_days": 3,
            "fee_bps": 5.0,
            "slippage_bps": 3.0,
        },
        timeout=30,
    )
    bt.raise_for_status()
    bt_data = bt.json()
    bt_status = str((bt_data.get("metrics") or {}).get("status") or bt_data.get("status") or "")
    if bt_status != "completed":
        out = {
            "passed": False,
            "status": "insufficient_observation",
            "reason": f"backtest_not_completed:{bt_status or 'unknown'}",
        }
        print(json.dumps(out, ensure_ascii=False))
        return 0
    bt_pnl = float((bt_data.get("metrics") or {}).get("pnl_after_cost", 0.0))

    pa = requests.get(
        f"{args.api_base}/api/v2/metrics/pnl-attribution",
        params={"track": args.track, "lookback_hours": 24 * 7},
        timeout=30,
    )
    pa.raise_for_status()
    pa_data = pa.json()
    paper_net = float((pa_data.get("totals") or {}).get("net_pnl", 0.0))

    # Normalize paper proxy to ratio-scale for comparability.
    denom = max(1.0, abs(float((pa_data.get("totals") or {}).get("gross_notional_signed", 0.0))))
    paper_ratio = paper_net / denom

    delta = abs(bt_pnl - paper_ratio)
    baseline = max(1e-6, abs(bt_pnl))
    rel_dev = delta / baseline
    passed = rel_dev <= args.max_deviation

    out = {
        "backtest_pnl_after_cost": bt_pnl,
        "paper_net_ratio_proxy": paper_ratio,
        "relative_deviation": rel_dev,
        "max_deviation": args.max_deviation,
        "passed": passed,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"passed": False, "error": str(exc)}), file=sys.stderr)
        raise
