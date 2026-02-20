#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _parse_targets(raw: str) -> List[str]:
    vals = [s.strip().upper() for s in str(raw or "").split(",") if s.strip()]
    return vals or ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "TRX", "AVAX", "LINK"]


def _required_metric_contract(metrics: Dict[str, Any]) -> List[str]:
    required = [
        "status",
        "pnl_after_cost",
        "max_drawdown",
        "sharpe_daily",
        "observation_days",
        "per_target",
        "cost_breakdown",
        "lineage_coverage",
        "alignment_audit",
        "leakage_checks",
    ]
    missing = [k for k in required if k not in (metrics or {})]
    if str((metrics or {}).get("status") or "").lower() != "completed":
        missing.append("status_completed")
    return missing


def _payload_from_args(args: argparse.Namespace, targets: List[str]) -> Dict[str, Any]:
    return {
        "track": args.track,
        "targets": targets,
        "run_source": "prod",
        "data_regime": "prod_live",
        "score_source": "model",
        "require_model_artifact": bool(args.require_model_artifact),
        "horizon": args.horizon,
        "model_name": args.model_name or None,
        "model_version": args.model_version or None,
        "data_version": args.data_version,
        "lookback_days": int(args.lookback_days),
        "train_days": int(args.train_days),
        "test_days": int(args.test_days),
        "fee_bps": float(args.fee_bps),
        "slippage_bps": float(args.slippage_bps),
        "signal_entry_z_min": float(args.signal_entry_z_min) if args.signal_entry_z_min is not None else None,
        "signal_exit_z_min": float(args.signal_exit_z_min) if args.signal_exit_z_min is not None else None,
        "position_max_weight_base": float(args.position_max_weight_base) if args.position_max_weight_base is not None else None,
        "position_max_weight_high_vol_mult": (
            float(args.position_max_weight_high_vol_mult)
            if args.position_max_weight_high_vol_mult is not None
            else None
        ),
        "cost_penalty_lambda": float(args.cost_penalty_lambda) if args.cost_penalty_lambda is not None else None,
        "signal_polarity_mode": args.signal_polarity_mode,
        "alignment_mode": args.alignment_mode,
        "alignment_version": args.alignment_version,
        "max_feature_staleness_hours": int(args.max_feature_staleness_hours),
    }


def _validate_cost_floor(fee_bps: float, slippage_bps: float) -> None:
    min_fee_bps = float(os.getenv("BACKTEST_MIN_FEE_BPS", "3.0"))
    min_slippage_bps = float(os.getenv("BACKTEST_MIN_SLIPPAGE_BPS", "2.0"))
    if float(fee_bps) < min_fee_bps or float(slippage_bps) < min_slippage_bps:
        raise RuntimeError(
            f"cost_params_too_low fee_bps={fee_bps} slippage_bps={slippage_bps} "
            f"(minimums: fee={min_fee_bps}, slippage={min_slippage_bps})"
        )


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


def _poll_task_result(
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
            err = str(payload.get("error") or "task_failed")
            raise RuntimeError(err)

        now = time.time()
        if now >= deadline:
            raise TimeoutError(f"task_max_wait_timeout status={status}")
        if now >= stall_deadline:
            raise TimeoutError(f"task_stalled status={status}")

        time.sleep(max(0.1, float(task_poll_sec)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run strict prod_live model backtest batch")
    ap.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    ap.add_argument("--track", default="liquid")
    ap.add_argument("--targets", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    ap.add_argument("--n-runs", type=int, default=24)
    ap.add_argument("--sleep-sec", type=float, default=0.2)
    ap.add_argument("--request-timeout-sec", type=float, default=12.0)
    ap.add_argument("--task-poll-sec", type=float, default=2.0)
    ap.add_argument("--task-max-wait-sec", type=float, default=1800.0)
    ap.add_argument("--task-stall-timeout-sec", type=float, default=300.0)
    ap.add_argument("--horizon", default="1d", choices=["1h", "1d", "7d"])
    ap.add_argument("--model-name", default=os.getenv("MODEL_NAME", "liquid_ttm_ensemble"))
    ap.add_argument("--model-version", default=os.getenv("MODEL_VERSION", "v2.1"))
    ap.add_argument("--data-version", default=os.getenv("DATA_VERSION", "v1"))
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--train-days", type=int, default=56)
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--fee-bps", type=float, default=float(os.getenv("COST_FEE_BPS", "5.0")))
    ap.add_argument("--slippage-bps", type=float, default=float(os.getenv("COST_SLIPPAGE_BPS", "3.0")))
    ap.add_argument("--signal-entry-z-min", type=float, default=None)
    ap.add_argument("--signal-exit-z-min", type=float, default=None)
    ap.add_argument("--position-max-weight-base", type=float, default=None)
    ap.add_argument("--position-max-weight-high-vol-mult", type=float, default=None)
    ap.add_argument("--cost-penalty-lambda", type=float, default=None)
    ap.add_argument(
        "--signal-polarity-mode",
        default=os.getenv("BACKTEST_SIGNAL_POLARITY_MODE", "auto_train_ic"),
        choices=["normal", "auto_train_ic", "auto_train_pnl"],
    )
    ap.add_argument("--alignment-mode", default=os.getenv("BACKTEST_ALIGNMENT_MODE", "strict_asof"), choices=["strict_asof"])
    ap.add_argument("--alignment-version", default=os.getenv("BACKTEST_ALIGNMENT_VERSION", "strict_asof_v1"))
    ap.add_argument("--max-feature-staleness-hours", type=int, default=int(os.getenv("BACKTEST_MAX_FEATURE_STALENESS_HOURS", str(24 * 14))))
    ap.add_argument("--require-model-artifact", action="store_true", default=True)
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    _validate_cost_floor(float(args.fee_bps), float(args.slippage_bps))

    started = datetime.now(timezone.utc)
    targets = _parse_targets(args.targets)
    payload = _payload_from_args(args, targets)

    attempted = 0
    completed = 0
    failed = 0
    contract_valid = 0
    contract_invalid = 0
    run_ids: List[int] = []
    task_ids: List[str] = []
    failures: List[Dict[str, Any]] = []
    contract_issues: List[Dict[str, Any]] = []

    api_base = args.api_base.rstrip("/")
    sess = _http_session()
    try:
        for i in range(max(1, int(args.n_runs))):
            attempted += 1
            try:
                task_id = _submit_backtest_task(
                    sess,
                    api_base=api_base,
                    payload=payload,
                    request_timeout_sec=float(args.request_timeout_sec),
                )
                task_ids.append(task_id)

                task_data = _poll_task_result(
                    sess,
                    api_base=api_base,
                    task_id=task_id,
                    request_timeout_sec=float(args.request_timeout_sec),
                    task_max_wait_sec=float(args.task_max_wait_sec),
                    task_stall_timeout_sec=float(args.task_stall_timeout_sec),
                    task_poll_sec=float(args.task_poll_sec),
                )
                body = (task_data or {}).get("result") or {}
                rid = int(body.get("run_id") or 0)
                if rid > 0:
                    run_ids.append(rid)

                metrics = (body or {}).get("metrics") or {}
                status = str((body or {}).get("status") or metrics.get("status") or "").lower()
                if status == "completed":
                    completed += 1
                else:
                    failed += 1
                    failures.append(
                        {
                            "idx": i,
                            "task_id": task_id,
                            "run_id": rid,
                            "status": status or "unknown",
                            "reason": metrics.get("reason"),
                        }
                    )

                missing = _required_metric_contract(metrics)
                if missing:
                    contract_invalid += 1
                    contract_issues.append({"idx": i, "task_id": task_id, "run_id": rid, "missing": missing})
                else:
                    contract_valid += 1
            except Exception as exc:
                failed += 1
                failures.append({"idx": i, "error": type(exc).__name__, "message": str(exc)[:240]})

            if i + 1 < max(1, int(args.n_runs)) and float(args.sleep_sec) > 0:
                time.sleep(float(args.sleep_sec))
    finally:
        sess.close()

    out = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "track": args.track,
        "api_base": args.api_base,
        "strict_filters": {
            "run_source": "prod",
            "score_source": "model",
            "data_regime": "prod_live",
        },
        "payload": payload,
        "attempted": attempted,
        "completed": completed,
        "failed": failed,
        "run_ids": run_ids,
        "task_ids": task_ids,
        "contract_valid": contract_valid,
        "contract_invalid": contract_invalid,
        "failures": failures[:30],
        "contract_issues": contract_issues[:30],
        "task_poll_sec": float(args.task_poll_sec),
        "task_max_wait_sec": float(args.task_max_wait_sec),
        "task_stall_timeout_sec": float(args.task_stall_timeout_sec),
    }
    print(json.dumps(out, ensure_ascii=False))

    if args.jsonl:
        p = Path(args.jsonl)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return 0 if completed > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
