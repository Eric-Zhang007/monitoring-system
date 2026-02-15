#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from statistics import mean, pstdev
from typing import Any, Dict, List


def _sharpe(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    sd = pstdev(vals)
    if sd <= 1e-12:
        return 0.0
    # approximate annualization for daily-like slices
    return float((mu / sd) * math.sqrt(252.0))


def _max_dd(vals: List[float]) -> float:
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    for r in vals:
        eq *= 1.0 + r
        peak = max(peak, eq)
        dd = 1.0 - (eq / peak)
        mdd = max(mdd, dd)
    return float(mdd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate codex-plan hard metrics")
    ap.add_argument("--db-url", default=os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor"))
    ap.add_argument("--track", default="liquid")
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--enforce", action="store_true")
    args = ap.parse_args()

    sql_backtest = (
        "SELECT COALESCE(metrics->>'pnl_after_cost','0') || ',' || "
        "COALESCE(metrics->>'max_drawdown','0') "
        "FROM backtest_runs "
        f"WHERE track='{args.track}' "
        f"AND created_at > NOW() - make_interval(days => {args.lookback_days}) "
        "AND metrics ? 'pnl_after_cost' "
        "AND metrics ? 'max_drawdown' "
        "ORDER BY created_at DESC LIMIT 500;"
    )
    cmd_bt = [
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
        sql_backtest,
    ]
    out_bt = subprocess.run(cmd_bt, capture_output=True, text=True, check=True).stdout.strip().splitlines()
    pnls: List[float] = []
    max_dds: List[float] = []
    for row in out_bt:
        val = row.strip()
        if not val:
            continue
        parts = val.split(",", 1)
        if len(parts) != 2:
            continue
        try:
            pnls.append(float(parts[0]))
            max_dds.append(float(parts[1]))
        except ValueError:
            continue

    sql_exec = (
        "SELECT COUNT(*) FILTER (WHERE status='rejected')::text || ',' || COUNT(*)::text "
        "FROM orders_sim "
        f"WHERE track='{args.track}' "
        f"AND created_at > NOW() - make_interval(days => {args.lookback_days});"
    )
    cmd_exec = [
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
        "-c",
        sql_exec,
    ]
    out_exec = subprocess.run(cmd_exec, capture_output=True, text=True, check=True).stdout.strip()
    rejected = 0
    total = 0
    if out_exec and "," in out_exec:
        a, b = out_exec.split(",", 1)
        rejected = int(float(a))
        total = int(float(b))

    sharpe = _sharpe(pnls)
    maxdd = max(max_dds) if max_dds else 0.0
    reject_rate = (rejected / total) if total > 0 else 0.0

    checks: Dict[str, Any] = {
        "oos_sharpe_gt_1_5": sharpe > 1.5,
        "maxdd_lt_0_12": maxdd < 0.12,
        "reject_rate_lt_0_01": reject_rate < 0.01,
    }
    passed = all(checks.values())
    out = {
        "track": args.track,
        "lookback_days": args.lookback_days,
        "samples": len(pnls),
        "sharpe": round(sharpe, 6),
        "max_drawdown": round(maxdd, 6),
        "execution_reject_rate": round(reject_rate, 6),
        "checks": checks,
        "passed": passed,
    }
    print(json.dumps(out, ensure_ascii=False))
    if args.enforce and not passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
