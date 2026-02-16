#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _vals(raw: str) -> List[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _score(metric: Dict[str, Any]) -> float:
    sharpe = float(metric.get("sharpe") or 0.0)
    pnl = float(metric.get("pnl_after_cost") or 0.0)
    maxdd = float(metric.get("max_drawdown") or 0.0)
    cov = float(metric.get("model_inference_coverage") or 0.0)
    return sharpe + 200.0 * pnl - 2.0 * maxdd + 0.1 * cov


def _http_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


def _submit_backtest_task(
    sess: requests.Session,
    *,
    api_base: str,
    payload: Dict[str, Any],
    request_timeout_sec: float,
) -> str:
    resp = sess.post(
        f"{api_base}/api/v2/tasks/backtest",
        json=payload,
        timeout=max(0.5, float(request_timeout_sec)),
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"submit_http_{resp.status_code}")
    body = resp.json() if resp.content else {}
    task_id = str(body.get("task_id") or "")
    if not task_id:
        raise RuntimeError("submit_missing_task_id")
    return task_id


def _poll_task(
    sess: requests.Session,
    *,
    api_base: str,
    task_id: str,
    request_timeout_sec: float,
    task_max_wait_sec: float,
    task_stall_timeout_sec: float,
    task_poll_sec: float,
) -> Dict[str, Any]:
    started = time.time()
    deadline = started + max(5.0, float(task_max_wait_sec))
    stall_deadline = started + max(5.0, float(task_stall_timeout_sec))
    last_status = ""

    while True:
        resp = sess.get(
            f"{api_base}/api/v2/tasks/{task_id}",
            timeout=max(0.5, float(request_timeout_sec)),
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"task_status_http_{resp.status_code}")

        payload = resp.json() if resp.content else {}
        status = str(payload.get("status") or "unknown").lower()
        if status != last_status:
            last_status = status
            stall_deadline = time.time() + max(5.0, float(task_stall_timeout_sec))

        if status == "completed":
            return payload
        if status == "failed":
            raise RuntimeError(str(payload.get("error") or "task_failed"))

        now = time.time()
        if now >= deadline:
            raise TimeoutError(f"task_max_wait_timeout status={status}")
        if now >= stall_deadline:
            raise TimeoutError(f"task_stalled status={status}")

        time.sleep(max(0.1, float(task_poll_sec)))


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
    ap.add_argument("--request-timeout-sec", type=float, default=12.0)
    ap.add_argument("--task-poll-sec", type=float, default=2.0)
    ap.add_argument("--task-max-wait-sec", type=float, default=1800.0)
    ap.add_argument("--task-stall-timeout-sec", type=float, default=300.0)
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
    api_base = args.api_base.rstrip("/")

    sess = _http_session()
    try:
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
                    task_id = _submit_backtest_task(
                        sess,
                        api_base=api_base,
                        payload=payload,
                        request_timeout_sec=float(args.request_timeout_sec),
                    )
                    row["task_id"] = task_id
                    task_data = _poll_task(
                        sess,
                        api_base=api_base,
                        task_id=task_id,
                        request_timeout_sec=float(args.request_timeout_sec),
                        task_max_wait_sec=float(args.task_max_wait_sec),
                        task_stall_timeout_sec=float(args.task_stall_timeout_sec),
                        task_poll_sec=float(args.task_poll_sec),
                    )
                    body = (task_data or {}).get("result") or {}
                    metric = (body or {}).get("metrics") or {}
                    row["status"] = str(metric.get("status") or body.get("status") or "unknown")
                    row["run_id"] = body.get("run_id")
                    row["metric"] = metric
                    if row["status"] == "completed":
                        row["score"] = _score(metric)
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = type(exc).__name__
                    row["error_message"] = str(exc)[:240]
                trials.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        sess.close()

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
                "task_id": t.get("task_id"),
            }
            for t in ok[:5]
        ],
        "task_poll_sec": float(args.task_poll_sec),
        "task_max_wait_sec": float(args.task_max_wait_sec),
        "task_stall_timeout_sec": float(args.task_stall_timeout_sec),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
