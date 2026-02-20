#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"optuna is required: {exc}")


@dataclass
class StageConfig:
    run_source: str
    data_regime: str
    lookback_days: int
    train_days: int
    test_days: int
    symbols: List[str]


def _default_workers(compute_tier: str) -> int:
    cpu_workers = max(1, multiprocessing.cpu_count() - 2)
    if compute_tier == "a100x2":
        return max(cpu_workers, 16)
    return cpu_workers


def _stage_config(stage: int, symbols: List[str]) -> StageConfig:
    syms = [s.strip().upper() for s in symbols if s.strip()]
    if not syms:
        syms = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "TRX", "AVAX", "LINK"]
    if stage == 1:
        return StageConfig(
            run_source="maintenance",
            data_regime="maintenance_replay",
            lookback_days=45,
            train_days=14,
            test_days=3,
            symbols=syms[:1],
        )
    if stage == 2:
        return StageConfig(
            run_source="maintenance",
            data_regime="maintenance_replay",
            lookback_days=120,
            train_days=28,
            test_days=7,
            symbols=syms,
        )
    return StageConfig(
        run_source="prod",
        data_regime="prod_live",
        lookback_days=240,
        train_days=56,
        test_days=14,
        symbols=syms,
    )


def _stage_default_hours(stage: int) -> float:
    if stage == 1:
        return 1.5
    if stage == 2:
        return 4.0
    return 8.0


def _estimate_cost_cny(
    *,
    compute_tier: str,
    estimated_hours: float,
    n_workers: int,
    a100_hourly_cny: float,
    cpu_hourly_cny: float,
    billing_discount: float,
) -> Dict[str, float]:
    workers = max(1, int(n_workers))
    cpu_units = float(min(16, workers))
    gpu_cards = 2.0 if compute_tier == "a100x2" else 0.0
    gpu_cost = float(estimated_hours) * float(a100_hourly_cny) * gpu_cards * float(max(0.0, billing_discount))
    cpu_cost = float(estimated_hours) * float(cpu_hourly_cny) * cpu_units * float(max(0.0, billing_discount))
    return {
        "gpu_cards": gpu_cards,
        "cpu_units": cpu_units,
        "gpu_cost_cny": round(gpu_cost, 2),
        "cpu_cost_cny": round(cpu_cost, 2),
        "total_cost_cny": round(gpu_cost + cpu_cost, 2),
    }


def _trial_score(metric: Dict[str, Any]) -> float:
    pnl = float(metric.get("pnl_after_cost") or 0.0)
    maxdd = float(metric.get("max_drawdown") or 1.0)
    reject = float(metric.get("execution_reject_rate") or 0.0)
    turnover = float(metric.get("turnover") or 0.0)
    # Scalarized objective: maximize pnl_after_cost while penalizing risk and turnover.
    return float(pnl - 2.5 * maxdd - 1.5 * reject - 0.02 * turnover)


def _run_backtest(api_base: str, payload: Dict[str, Any], timeout_sec: float) -> Dict[str, Any]:
    resp = requests.post(f"{api_base}/api/v2/backtest/run", json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    body = resp.json() if resp.content else {}
    out = (body or {}).get("metrics") or {}
    out["run_id"] = body.get("run_id")
    out["status"] = body.get("status") or out.get("status")
    return out


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Parallel Optuna HPO for liquid strategy")
    ap.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    ap.add_argument("--track", default="liquid")
    ap.add_argument("--symbols", default="BTC,ETH,SOL")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--request-timeout-sec", type=float, default=45.0)
    ap.add_argument("--storage", default="artifacts/hpo/optuna_liquid.db")
    ap.add_argument("--study-name", default="liquid_hpo")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compute-tier", default="local", choices=["local", "a100x2"])
    ap.add_argument("--max-runtime-min", type=float, default=0.0)
    ap.add_argument("--estimated-hours", type=float, default=0.0)
    ap.add_argument("--a100-hourly-cny", type=float, default=float(os.getenv("A100_HOURLY_CNY", "11.96")))
    ap.add_argument("--cpu-hourly-cny", type=float, default=float(os.getenv("CPU_HOURLY_CNY", "0.0")))
    ap.add_argument("--billing-mode", default=os.getenv("AUTODL_BILLING_MODE", "hourly"), choices=["hourly", "daily", "monthly"])
    ap.add_argument("--daily-discount", type=float, default=float(os.getenv("AUTODL_DAILY_DISCOUNT", "0.92")))
    ap.add_argument("--monthly-discount", type=float, default=float(os.getenv("AUTODL_MONTHLY_DISCOUNT", "0.85")))
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    stage_cfg = _stage_config(int(args.stage), symbols)
    n_workers = int(args.n_workers) if int(args.n_workers) > 0 else _default_workers(args.compute_tier)
    estimated_hours = float(args.estimated_hours) if float(args.estimated_hours) > 0 else _stage_default_hours(int(args.stage))

    storage_path = Path(args.storage)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_uri = f"sqlite:///{storage_path.as_posix()}"
    study_name = args.study_name if args.resume else f"{args.study_name}_s{args.stage}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    trial_log = storage_path.parent / f"optuna_trials_{study_name}.jsonl"

    sampler = TPESampler(seed=42, multivariate=True)
    pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=0, interval_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=bool(args.resume),
    )

    started = time.time()

    def _objective(trial: optuna.trial.Trial) -> float:
        entry_z = trial.suggest_float("signal_entry_z_min", 0.008, 0.06, log=True)
        exit_z = trial.suggest_float("signal_exit_z_min", 0.002, min(entry_z * 0.8, 0.03))
        base_w = trial.suggest_float("position_max_weight_base", 0.02, 0.22)
        high_vol_mult = trial.suggest_float("position_max_weight_high_vol_mult", 0.2, 1.0)
        cost_lambda = trial.suggest_float("cost_penalty_lambda", 0.2, 3.0)
        fee_bps = trial.suggest_float("fee_bps", 3.0, 8.0)
        slippage_bps = trial.suggest_float("slippage_bps", 2.0, 8.0)

        payload = {
            "track": args.track,
            "run_source": stage_cfg.run_source,
            "data_regime": stage_cfg.data_regime,
            "score_source": "model",
            "require_model_artifact": True,
            "targets": stage_cfg.symbols,
            "horizon": "1d",
            "data_version": os.getenv("DATA_VERSION", "v1"),
            "lookback_days": stage_cfg.lookback_days,
            "train_days": stage_cfg.train_days,
            "test_days": stage_cfg.test_days,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "signal_entry_z_min": entry_z,
            "signal_exit_z_min": exit_z,
            "position_max_weight_base": base_w,
            "position_max_weight_high_vol_mult": high_vol_mult,
            "cost_penalty_lambda": cost_lambda,
        }

        metric: Dict[str, Any]
        try:
            metric = _run_backtest(args.api_base, payload, timeout_sec=float(args.request_timeout_sec))
        except Exception as exc:
            trial.set_user_attr("error", type(exc).__name__)
            score = -1e6
            _append_jsonl(
                trial_log,
                {
                    "study": study_name,
                    "trial": trial.number,
                    "status": "error",
                    "error": type(exc).__name__,
                    "params": payload,
                    "score": score,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            )
            return score

        status = str(metric.get("status") or "unknown")
        if status != "completed":
            score = -5e5
        else:
            score = _trial_score(metric)

        trial.set_user_attr("run_id", metric.get("run_id"))
        trial.set_user_attr("status", status)
        trial.set_user_attr("score", score)

        _append_jsonl(
            trial_log,
            {
                "study": study_name,
                "trial": trial.number,
                "status": status,
                "score": score,
                "params": payload,
                "metric": metric,
                "ts": datetime.now(timezone.utc).isoformat(),
            },
        )

        if args.max_runtime_min > 0 and (time.time() - started) / 60.0 >= float(args.max_runtime_min):
            study.stop()

        return score

    study.optimize(_objective, n_trials=int(args.n_trials), n_jobs=max(1, n_workers), gc_after_trial=True)

    best = study.best_trial if study.best_trials else None
    discount = 1.0
    if args.billing_mode == "daily":
        discount = float(args.daily_discount)
    elif args.billing_mode == "monthly":
        discount = float(args.monthly_discount)
    out = {
        "study": study_name,
        "storage": storage_uri,
        "stage": int(args.stage),
        "compute_tier": args.compute_tier,
        "n_trials": int(args.n_trials),
        "n_workers": int(max(1, n_workers)),
        "trial_log": str(trial_log),
        "best_trial": {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "attrs": best.user_attrs,
        }
        if best is not None
        else None,
        "cost_estimate": {
            "currency": "CNY",
            "estimated_hours": round(float(estimated_hours), 3),
            "inputs": {
                "compute_tier": args.compute_tier,
                "n_workers": int(max(1, n_workers)),
                "a100_hourly_cny": float(args.a100_hourly_cny),
                "cpu_hourly_cny": float(args.cpu_hourly_cny),
                "billing_mode": args.billing_mode,
                "billing_discount": float(discount),
            },
            "estimate": _estimate_cost_cny(
                compute_tier=args.compute_tier,
                estimated_hours=estimated_hours,
                n_workers=max(1, n_workers),
                a100_hourly_cny=float(args.a100_hourly_cny),
                cpu_hourly_cny=float(args.cpu_hourly_cny),
                billing_discount=float(discount),
            ),
        },
        "cost_plan": {
            "stage_1": {
                "objective": "coarse_search",
                "recommended_compute_tier": "local",
                "default_hours": _stage_default_hours(1),
            },
            "stage_2": {
                "objective": "multi_symbol_refine",
                "recommended_compute_tier": "local",
                "default_hours": _stage_default_hours(2),
            },
            "stage_3": {
                "objective": "prod_oos_confirmation",
                "recommended_compute_tier": "a100x2",
                "default_hours": _stage_default_hours(3),
            },
        },
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
