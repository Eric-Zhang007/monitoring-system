#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Any, Dict, List, Tuple

import requests


def _run_backtest(api_base: str, payload: Dict[str, Any], timeout_sec: float) -> Dict[str, Any]:
    r = requests.post(f"{api_base}/api/v2/backtest/run", json=payload, timeout=timeout_sec)
    r.raise_for_status()
    body = r.json()
    metrics = (body or {}).get("metrics") or {}
    metrics["run_id"] = body.get("run_id")
    metrics["status"] = body.get("status") or metrics.get("status")
    return metrics


def _parse_grid_values(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _trade_proxy(metric: Dict[str, Any]) -> float:
    explicit = _safe_float(metric.get("trades"), default=-1.0)
    if explicit >= 0.0:
        return explicit
    turnover = _safe_float(metric.get("turnover"), default=0.0)
    samples = _safe_float(metric.get("samples"), default=0.0)
    return max(0.0, turnover * samples)


def _active_target_count(metric: Dict[str, Any], min_abs_target_pnl: float) -> int:
    per_target = metric.get("per_target")
    if not isinstance(per_target, dict):
        return 0
    active = 0
    for sub in per_target.values():
        if not isinstance(sub, dict):
            continue
        pnl = abs(_safe_float(sub.get("pnl_after_cost"), default=0.0))
        turnover = _safe_float(sub.get("turnover"), default=0.0)
        if pnl >= min_abs_target_pnl or turnover > 0.0:
            active += 1
    return active


def _activity_checks(
    metric: Dict[str, Any],
    *,
    min_turnover: float,
    min_trades: float,
    min_abs_pnl: float,
    min_active_targets: int,
) -> Tuple[bool, Dict[str, Any]]:
    turnover = _safe_float(metric.get("turnover"), default=0.0)
    trades = _trade_proxy(metric)
    abs_pnl = abs(_safe_float(metric.get("pnl_after_cost"), default=0.0))
    active_targets = _active_target_count(metric, min_abs_target_pnl=max(1e-8, float(min_abs_pnl) / 4.0))
    checks = {
        "turnover_ge_min": turnover >= float(min_turnover),
        "trades_ge_min": trades >= float(min_trades),
        "abs_pnl_ge_min": abs_pnl >= float(min_abs_pnl),
        "active_targets_ge_min": active_targets >= int(min_active_targets),
    }
    return all(checks.values()), {
        "turnover": turnover,
        "trade_proxy": trades,
        "abs_pnl_after_cost": abs_pnl,
        "active_targets": active_targets,
        "checks": checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Grid tune liquid strategy params via /api/v2/backtest/run")
    ap.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--run-source", default="prod", choices=["prod", "maintenance"])
    ap.add_argument("--data-regime", default="", help="prod_live|maintenance_replay|mixed; empty means infer from run-source")
    ap.add_argument("--score-source", default="model", choices=["model", "heuristic"])
    ap.add_argument("--signal-polarity-mode", default="auto_train_ic", choices=["normal", "auto_train_ic", "auto_train_pnl"])
    ap.add_argument("--alignment-mode", default="strict_asof", choices=["strict_asof", "legacy_index"])
    ap.add_argument("--alignment-version", default="strict_asof_v1")
    ap.add_argument("--max-feature-staleness-hours", type=int, default=24 * 14)
    ap.add_argument("--targets", default="BTC,ETH,SOL", help="comma-separated targets")
    ap.add_argument("--data-version", default="v1")
    ap.add_argument("--fee-bps", type=float, default=0.5)
    ap.add_argument("--slippage-bps", type=float, default=0.2)
    ap.add_argument("--train-days", type=int, default=35)
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--max-trials", type=int, default=512)
    ap.add_argument("--request-timeout-sec", type=float, default=45.0)
    ap.add_argument("--entry-grid", default="0.08,0.06,0.05,0.04,0.03,0.02,0.015")
    ap.add_argument("--exit-grid", default="0.005,0.008,0.01,0.012,0.015,0.02")
    ap.add_argument("--base-weight-grid", default="0.08,0.1,0.12,0.15,0.18")
    ap.add_argument("--high-vol-mult-grid", default="0.35,0.45,0.55,0.65,0.75")
    ap.add_argument("--cost-lambda-grid", default="0.8,1.0,1.2,1.5,1.8,2.2")
    ap.add_argument("--min-turnover", type=float, default=0.05)
    ap.add_argument("--min-trades", type=float, default=5.0)
    ap.add_argument("--min-abs-pnl", type=float, default=1e-5)
    ap.add_argument("--min-active-targets", type=int, default=2)
    ap.add_argument("--max-reject-rate", type=float, default=0.02)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    targets = [s.strip().upper() for s in str(args.targets).split(",") if s.strip()]
    inferred_data_regime = str(args.data_regime or "").strip().lower()
    if not inferred_data_regime:
        inferred_data_regime = "prod_live" if args.run_source == "prod" else "maintenance_replay"

    grid = {
        "SIGNAL_ENTRY_Z_MIN": _parse_grid_values(args.entry_grid),
        "SIGNAL_EXIT_Z_MIN": _parse_grid_values(args.exit_grid),
        "POSITION_MAX_WEIGHT_BASE": _parse_grid_values(args.base_weight_grid),
        "POSITION_MAX_WEIGHT_HIGH_VOL_MULT": _parse_grid_values(args.high_vol_mult_grid),
        "COST_PENALTY_LAMBDA": _parse_grid_values(args.cost_lambda_grid),
    }

    keys = list(grid.keys())
    trials = []
    for values in itertools.product(*(grid[k] for k in keys)):
        if len(trials) >= args.max_trials:
            break
        cfg = dict(zip(keys, values))
        payload = {
            "track": "liquid",
            "run_source": args.run_source,
            "data_regime": inferred_data_regime,
            "score_source": args.score_source,
            "signal_polarity_mode": args.signal_polarity_mode,
            "alignment_mode": args.alignment_mode,
            "alignment_version": args.alignment_version,
            "max_feature_staleness_hours": int(args.max_feature_staleness_hours),
            "require_model_artifact": True,
            "targets": targets,
            "horizon": "1d",
            "data_version": args.data_version,
            "lookback_days": int(args.lookback_days),
            "train_days": int(args.train_days),
            "test_days": int(args.test_days),
            "fee_bps": float(args.fee_bps),
            "slippage_bps": float(args.slippage_bps),
            "signal_entry_z_min": float(cfg["SIGNAL_ENTRY_Z_MIN"]),
            "signal_exit_z_min": float(cfg["SIGNAL_EXIT_Z_MIN"]),
            "position_max_weight_base": float(cfg["POSITION_MAX_WEIGHT_BASE"]),
            "position_max_weight_high_vol_mult": float(cfg["POSITION_MAX_WEIGHT_HIGH_VOL_MULT"]),
            "cost_penalty_lambda": float(cfg["COST_PENALTY_LAMBDA"]),
        }
        try:
            metric = _run_backtest(args.api_base, payload, timeout_sec=float(args.request_timeout_sec))
        except Exception as exc:
            trials.append({"config": cfg, "status": "error", "error": type(exc).__name__})
            continue
        if str(metric.get("status")) != "completed":
            trials.append(
                {
                    "config": cfg,
                    "status": str(metric.get("status") or "failed"),
                    "reason": str(metric.get("reason") or ""),
                    "run_id": metric.get("run_id"),
                }
            )
            continue
        trials.append(
            {
                "config": cfg,
                "payload": {
                    "fee_bps": float(args.fee_bps),
                    "slippage_bps": float(args.slippage_bps),
                    "signal_polarity_mode": args.signal_polarity_mode,
                    "alignment_mode": args.alignment_mode,
                    "alignment_version": args.alignment_version,
                    "max_feature_staleness_hours": int(args.max_feature_staleness_hours),
                },
                "metric": metric,
                "status": "ok",
            }
        )

    def _score(item: Dict[str, Any]) -> tuple[float, float, float]:
        m = item.get("metric") or {}
        sharpe = _safe_float(m.get("sharpe_daily"), default=_safe_float(m.get("sharpe"), default=0.0))
        maxdd = _safe_float(m.get("max_drawdown"), default=1.0)
        reject = _safe_float(m.get("execution_reject_rate"), default=1.0)
        turnover = _safe_float(m.get("turnover"), default=0.0)
        penalty = 0.0
        if maxdd >= 0.12:
            penalty += 10.0 + maxdd
        if reject >= float(args.max_reject_rate):
            penalty += 10.0 + reject
        if turnover <= 0.0:
            penalty += 25.0
        return (sharpe - penalty, -maxdd, -reject)

    activity_rejected: List[Dict[str, Any]] = []
    ok_rows: List[Dict[str, Any]] = []
    for row in trials:
        if row.get("status") != "ok":
            continue
        metric = row.get("metric") or {}
        active_ok, details = _activity_checks(
            metric,
            min_turnover=float(args.min_turnover),
            min_trades=float(args.min_trades),
            min_abs_pnl=float(args.min_abs_pnl),
            min_active_targets=int(args.min_active_targets),
        )
        row["activity"] = details
        if not active_ok:
            row["status"] = "inactive_rejected"
            activity_rejected.append(
                {
                    "config": row.get("config"),
                    "activity": details,
                    "run_id": metric.get("run_id"),
                }
            )
            continue
        ok_rows.append(row)

    ok_rows.sort(key=_score, reverse=True)
    out = {
        "grid_size": int(
            len(grid["SIGNAL_ENTRY_Z_MIN"])
            * len(grid["SIGNAL_EXIT_Z_MIN"])
            * len(grid["POSITION_MAX_WEIGHT_BASE"])
            * len(grid["POSITION_MAX_WEIGHT_HIGH_VOL_MULT"])
            * len(grid["COST_PENALTY_LAMBDA"])
        ),
        "trials": len(trials),
        "ok_trials_before_activity_filter": len([t for t in trials if t.get("status") in {"ok", "inactive_rejected"}]),
        "ok_trials": len(ok_rows),
        "inactive_rejected_trials": len(activity_rejected),
        "activity_constraints": {
            "min_turnover": float(args.min_turnover),
            "min_trades": float(args.min_trades),
            "min_abs_pnl": float(args.min_abs_pnl),
            "min_active_targets": int(args.min_active_targets),
            "max_reject_rate": float(args.max_reject_rate),
        },
        "best": ok_rows[: max(1, int(args.top_k))],
        "inactive_examples": activity_rejected[:20],
        "errors_or_failed": [t for t in trials if t.get("status") not in {"ok", "inactive_rejected"}][:20],
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0 if ok_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
