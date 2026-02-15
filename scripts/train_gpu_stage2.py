#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": int(p.returncode),
        "stdout": p.stdout[-4000:],
        "stderr": p.stderr[-4000:],
    }


def _estimate_cost_cny(hours: float, compute_tier: str, a100_hourly_cny: float, billing_discount: float) -> float:
    if compute_tier == "a100x2":
        return round(float(hours) * float(a100_hourly_cny) * 2.0 * float(max(0.0, billing_discount)), 2)
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage-2 GPU training orchestrator (strict-asof + external events)")
    ap.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    ap.add_argument("--epochs", type=int, default=int(os.getenv("LIQUID_EPOCHS", "24")))
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("LIQUID_BATCH_SIZE", "128")))
    ap.add_argument("--compute-tier", default=os.getenv("COMPUTE_TIER", "local"), choices=["local", "a100x2"])
    ap.add_argument("--run-optuna", action="store_true")
    ap.add_argument("--optuna-trials", type=int, default=80)
    ap.add_argument("--enable-vc", action="store_true")
    ap.add_argument("--enable-liquid", action="store_true", default=True)
    ap.add_argument("--a100-hourly-cny", type=float, default=float(os.getenv("A100_HOURLY_CNY", "11.96")))
    ap.add_argument("--billing-discount", type=float, default=float(os.getenv("AUTODL_BILLING_DISCOUNT", "1.0")))
    ap.add_argument("--estimated-hours", type=float, default=6.0)
    ap.add_argument("--out", default="artifacts/gpu_stage2/train_gpu_stage2_latest.json")
    args = ap.parse_args()

    started = datetime.now(timezone.utc)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LIQUID_SYMBOLS"] = str(args.symbols)
    env["EPOCHS"] = str(int(args.epochs))
    env["BATCH_SIZE"] = str(int(args.batch_size))
    env["TRAIN_RUN_ONCE"] = "1"
    env["TRAIN_ENABLE_VC"] = "1" if bool(args.enable_vc) else "0"
    env["TRAIN_ENABLE_LIQUID"] = "1" if bool(args.enable_liquid) else "0"
    env["BACKTEST_ALIGNMENT_MODE"] = "strict_asof"
    env["BACKTEST_ALIGNMENT_VERSION"] = "strict_asof_v1"
    env["BACKTEST_MAX_FEATURE_STALENESS_HOURS"] = str(24 * 14)

    steps: List[Dict[str, Any]] = []
    if bool(args.run_optuna):
        steps.append(
            _run(
                [
                    "python3",
                    "scripts/optuna_liquid_hpo.py",
                    "--compute-tier",
                    str(args.compute_tier),
                    "--n-trials",
                    str(int(args.optuna_trials)),
                    "--a100-hourly-cny",
                    str(float(args.a100_hourly_cny)),
                ],
                env=env,
            )
        )

    steps.append(_run(["python3", "training/main.py"], env=env))

    ok = all(int(s.get("returncode", 1)) == 0 for s in steps)
    finished = datetime.now(timezone.utc)
    total_cost = _estimate_cost_cny(
        hours=float(args.estimated_hours),
        compute_tier=str(args.compute_tier),
        a100_hourly_cny=float(args.a100_hourly_cny),
        billing_discount=float(args.billing_discount),
    )
    out = {
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_sec": round((finished - started).total_seconds(), 3),
        "status": "ok" if ok else "failed",
        "compute_tier": str(args.compute_tier),
        "symbols": [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()],
        "estimated_hours": float(args.estimated_hours),
        "cost_estimate_cny": float(total_cost),
        "steps": steps,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
