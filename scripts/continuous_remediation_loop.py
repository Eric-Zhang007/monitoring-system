#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _run_json(cmd: list[str]) -> Tuple[int, Dict[str, Any], str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    obj: Dict[str, Any] = {}
    if out:
        for line in reversed(out.splitlines()):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                break
            except Exception:
                continue
    return p.returncode, obj, out, err


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _parse_targets(raw: str) -> List[str]:
    out = [s.strip().upper() for s in str(raw or "").split(",") if s.strip()]
    return out or ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "TRX", "AVAX", "LINK"]


def _metric_trade_proxy(metric: Dict[str, Any]) -> float:
    explicit = _safe_float(metric.get("trades"), default=-1.0)
    if explicit >= 0.0:
        return explicit
    return max(0.0, _safe_float(metric.get("turnover"), default=0.0) * _safe_float(metric.get("samples"), default=0.0))


def _metric_active_targets(metric: Dict[str, Any], min_abs_target_pnl: float) -> int:
    per_target = metric.get("per_target")
    if not isinstance(per_target, dict):
        return 0
    cnt = 0
    for sub in per_target.values():
        if not isinstance(sub, dict):
            continue
        pnl = abs(_safe_float(sub.get("pnl_after_cost"), default=0.0))
        turnover = _safe_float(sub.get("turnover"), default=0.0)
        if pnl >= min_abs_target_pnl or turnover > 0.0:
            cnt += 1
    return cnt


def _metric_is_active(
    metric: Dict[str, Any],
    *,
    min_turnover: float,
    min_trades: float,
    min_abs_pnl: float,
    min_active_targets: int,
) -> Tuple[bool, Dict[str, Any]]:
    turnover = _safe_float(metric.get("turnover"), default=0.0)
    trades = _metric_trade_proxy(metric)
    abs_pnl = abs(_safe_float(metric.get("pnl_after_cost"), default=0.0))
    active_targets = _metric_active_targets(metric, min_abs_target_pnl=max(1e-8, float(min_abs_pnl) / 4.0))
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


def _candidate_signature(c: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float, float]:
    return (
        round(_safe_float(c.get("signal_entry_z_min"), default=0.0), 6),
        round(_safe_float(c.get("signal_exit_z_min"), default=0.0), 6),
        round(_safe_float(c.get("position_max_weight_base"), default=0.0), 6),
        round(_safe_float(c.get("position_max_weight_high_vol_mult"), default=0.0), 6),
        round(_safe_float(c.get("cost_penalty_lambda"), default=0.0), 6),
        round(_safe_float(c.get("fee_bps"), default=0.0), 6),
        round(_safe_float(c.get("slippage_bps"), default=0.0), 6),
        1.0 if str(c.get("signal_polarity_mode") or "").strip().lower() == "auto_train_ic" else 0.0,
    )


def _candidate_sort_key(c: Dict[str, Any]) -> Tuple[float, float, float]:
    score = _safe_float(c.get("score"), default=-1e9)
    abs_pnl = abs(_safe_float(c.get("pnl_after_cost"), default=0.0))
    turnover = _safe_float(c.get("turnover"), default=0.0)
    return (score, abs_pnl, turnover)


def _dedup_candidates(candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    uniq: Dict[Tuple[float, float, float, float, float, float, float, float], Dict[str, Any]] = {}
    for c in candidates:
        sig = _candidate_signature(c)
        prev = uniq.get(sig)
        if prev is None or _candidate_sort_key(c) > _candidate_sort_key(prev):
            uniq[sig] = c
    rows = list(uniq.values())
    rows.sort(key=_candidate_sort_key, reverse=True)
    return rows[: max(1, int(top_k))]


def _grid_item_to_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    cfg = item.get("config") if isinstance(item.get("config"), dict) else {}
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    metric = item.get("metric") if isinstance(item.get("metric"), dict) else {}
    c = {
        "source": "grid",
        "signal_entry_z_min": _safe_float(cfg.get("SIGNAL_ENTRY_Z_MIN"), default=0.0),
        "signal_exit_z_min": _safe_float(cfg.get("SIGNAL_EXIT_Z_MIN"), default=0.0),
        "position_max_weight_base": _safe_float(cfg.get("POSITION_MAX_WEIGHT_BASE"), default=0.0),
        "position_max_weight_high_vol_mult": _safe_float(cfg.get("POSITION_MAX_WEIGHT_HIGH_VOL_MULT"), default=0.0),
        "cost_penalty_lambda": _safe_float(cfg.get("COST_PENALTY_LAMBDA"), default=0.0),
        "fee_bps": _safe_float(payload.get("fee_bps"), default=_safe_float(metric.get("fee_bps"), default=0.0)),
        "slippage_bps": _safe_float(payload.get("slippage_bps"), default=_safe_float(metric.get("slippage_bps"), default=0.0)),
        "signal_polarity_mode": str(payload.get("signal_polarity_mode") or "auto_train_ic"),
        "score": _safe_float(metric.get("sharpe_daily"), default=_safe_float(metric.get("sharpe"), default=0.0)),
        "pnl_after_cost": _safe_float(metric.get("pnl_after_cost"), default=0.0),
        "turnover": _safe_float(metric.get("turnover"), default=0.0),
        "run_id": metric.get("run_id"),
    }
    return c


def _optuna_row_to_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    params = row.get("params") if isinstance(row.get("params"), dict) else {}
    metric = row.get("metric") if isinstance(row.get("metric"), dict) else {}
    c = {
        "source": "optuna",
        "signal_entry_z_min": _safe_float(params.get("signal_entry_z_min"), default=0.0),
        "signal_exit_z_min": _safe_float(params.get("signal_exit_z_min"), default=0.0),
        "position_max_weight_base": _safe_float(params.get("position_max_weight_base"), default=0.0),
        "position_max_weight_high_vol_mult": _safe_float(params.get("position_max_weight_high_vol_mult"), default=0.0),
        "cost_penalty_lambda": _safe_float(params.get("cost_penalty_lambda"), default=0.0),
        "fee_bps": _safe_float(params.get("fee_bps"), default=0.0),
        "slippage_bps": _safe_float(params.get("slippage_bps"), default=0.0),
        "signal_polarity_mode": str(params.get("signal_polarity_mode") or "auto_train_ic"),
        "score": _safe_float(row.get("score"), default=-1e9),
        "pnl_after_cost": _safe_float(metric.get("pnl_after_cost"), default=0.0),
        "turnover": _safe_float(metric.get("turnover"), default=0.0),
        "run_id": metric.get("run_id"),
    }
    return c


def _load_candidates_from_file(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not path.exists():
        return [], {"status": "missing", "path": str(path)}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], {"status": "error", "path": str(path), "error": type(exc).__name__}
    if isinstance(obj, dict):
        if isinstance(obj.get("candidates"), list):
            rows = [x for x in obj.get("candidates") if isinstance(x, dict)]
            return rows, {"status": "ok", "path": str(path), "count": len(rows), "shape": "dict.candidates"}
        if isinstance(obj.get("best"), list):
            rows = [_grid_item_to_candidate(x) for x in obj.get("best") if isinstance(x, dict)]
            return rows, {"status": "ok", "path": str(path), "count": len(rows), "shape": "dict.best"}
    if isinstance(obj, list):
        rows = [x for x in obj if isinstance(x, dict)]
        return rows, {"status": "ok", "path": str(path), "count": len(rows), "shape": "list"}
    return [], {"status": "empty", "path": str(path)}


def _load_candidates_from_optuna_logs(
    log_glob: str,
    *,
    min_turnover: float,
    min_trades: float,
    min_abs_pnl: float,
    min_active_targets: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    files = sorted(glob.glob(log_glob), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    if not files:
        return [], {"status": "missing", "glob": log_glob}
    chosen = Path(files[0])
    rows = []
    parse_errors = 0
    with chosen.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                parse_errors += 1
    candidates: List[Dict[str, Any]] = []
    rejected = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").lower()
        if status not in {"completed", "ok"}:
            continue
        metric = row.get("metric") if isinstance(row.get("metric"), dict) else {}
        if metric and str(metric.get("status") or "").lower() not in {"completed", ""}:
            continue
        if metric:
            active_ok, _ = _metric_is_active(
                metric,
                min_turnover=min_turnover,
                min_trades=min_trades,
                min_abs_pnl=min_abs_pnl,
                min_active_targets=min_active_targets,
            )
            if not active_ok:
                rejected += 1
                continue
        candidates.append(_optuna_row_to_candidate(row))
    meta = {
        "status": "ok",
        "glob": log_glob,
        "file": str(chosen),
        "rows": len(rows),
        "parse_errors": parse_errors,
        "candidates": len(candidates),
        "activity_rejected": rejected,
    }
    return candidates, meta


def _discover_candidates(args: argparse.Namespace, targets: List[str], it_dir: Path) -> Dict[str, Any]:
    source = str(args.candidate_source).strip().lower()
    top_k = max(1, int(args.candidate_top_k))
    logs: List[Dict[str, Any]] = []
    pool: List[Dict[str, Any]] = []

    if source in {"file", "auto"} and str(args.candidate_file).strip():
        file_rows, info = _load_candidates_from_file(Path(str(args.candidate_file).strip()))
        logs.append({"source": "file", **info})
        pool.extend(file_rows)

    if (not pool) and source in {"optuna", "auto"}:
        opt_rows, info = _load_candidates_from_optuna_logs(
            str(args.candidate_optuna_log_glob),
            min_turnover=float(args.candidate_min_turnover),
            min_trades=float(args.candidate_min_trades),
            min_abs_pnl=float(args.candidate_min_abs_pnl),
            min_active_targets=int(args.candidate_min_active_targets),
        )
        logs.append({"source": "optuna", **info})
        pool.extend(opt_rows)

    if (not pool) and source in {"grid", "auto"}:
        cmd = [
            "python3",
            "scripts/tune_liquid_strategy_grid.py",
            "--api-base",
            str(args.api_base),
            "--run-source",
            "maintenance",
            "--data-regime",
            "maintenance_replay",
            "--score-source",
            "model",
            "--signal-polarity-mode",
            str(args.signal_polarity_mode),
            "--fee-bps",
            str(float(args.fee_bps)),
            "--slippage-bps",
            str(float(args.slippage_bps)),
            "--targets",
            ",".join(targets),
            "--lookback-days",
            "180",
            "--train-days",
            "56",
            "--test-days",
            "14",
            "--max-trials",
            str(max(1, int(args.candidate_grid_max_trials))),
            "--request-timeout-sec",
            str(float(args.candidate_grid_timeout_sec)),
            "--min-turnover",
            str(float(args.candidate_min_turnover)),
            "--min-trades",
            str(float(args.candidate_min_trades)),
            "--min-abs-pnl",
            str(float(args.candidate_min_abs_pnl)),
            "--min-active-targets",
            str(max(1, int(args.candidate_min_active_targets))),
            "--max-reject-rate",
            str(float(args.candidate_grid_max_reject_rate)),
            "--top-k",
            str(top_k),
        ]
        rc, obj, out, err = _run_json(cmd)
        grid_rows = []
        if rc == 0 and isinstance(obj, dict):
            best = obj.get("best") if isinstance(obj.get("best"), list) else []
            grid_rows = [_grid_item_to_candidate(x) for x in best if isinstance(x, dict)]
        logs.append(
            {
                "source": "grid",
                "cmd": cmd,
                "exit_code": rc,
                "candidates": len(grid_rows),
                "stdout_tail": out[-500:] if out else "",
                "stderr_tail": err[-500:] if err else "",
            }
        )
        pool.extend(grid_rows)
        _write_json(it_dir / "candidate_discovery_grid.json", {"cmd": cmd, "exit_code": rc, "json": obj, "stderr": err})

    deduped = _dedup_candidates(pool, top_k=top_k)
    return {
        "source_mode": source,
        "requested_top_k": top_k,
        "targets": targets,
        "discovery_logs": logs,
        "candidates_found": len(pool),
        "candidates_selected": len(deduped),
        "candidates": deduped,
    }


def _candidate_cli_args(candidate: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    pairs = [
        ("--signal-entry-z-min", "signal_entry_z_min"),
        ("--signal-exit-z-min", "signal_exit_z_min"),
        ("--position-max-weight-base", "position_max_weight_base"),
        ("--position-max-weight-high-vol-mult", "position_max_weight_high_vol_mult"),
        ("--cost-penalty-lambda", "cost_penalty_lambda"),
        ("--fee-bps", "fee_bps"),
        ("--slippage-bps", "slippage_bps"),
    ]
    for flag, key in pairs:
        val = _safe_float(candidate.get(key), default=0.0)
        if val > 0.0:
            out.extend([flag, str(val)])
    mode = str(candidate.get("signal_polarity_mode") or "").strip().lower()
    if mode in {"normal", "auto_train_ic", "auto_train_pnl"}:
        out.extend(["--signal-polarity-mode", mode])
    return out


def _fallback_candidate(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "source": "fallback",
        "signal_entry_z_min": float(getattr(args, "fallback_entry_z", 0.08)),
        "signal_exit_z_min": float(getattr(args, "fallback_exit_z", 0.028)),
        "position_max_weight_base": float(getattr(args, "fallback_base_weight", 0.08)),
        "position_max_weight_high_vol_mult": float(getattr(args, "fallback_high_vol_mult", 0.35)),
        "cost_penalty_lambda": float(getattr(args, "fallback_cost_lambda", 1.0)),
        "fee_bps": float(getattr(args, "fee_bps", 0.5)),
        "slippage_bps": float(getattr(args, "slippage_bps", 0.2)),
        "signal_polarity_mode": str(getattr(args, "signal_polarity_mode", "auto_train_ic")),
        "score": 0.0,
        "pnl_after_cost": 0.0,
        "turnover": 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Continuous remediation loop: review -> patch inputs -> test -> gate")
    ap.add_argument("--api-base", default="http://localhost:8000")
    ap.add_argument("--max-iterations", type=int, default=0, help="0 means no hard limit")
    ap.add_argument("--green-windows", type=int, default=3)
    ap.add_argument("--sleep-sec", type=float, default=5.0)
    ap.add_argument("--batch-runs", type=int, default=6)
    ap.add_argument("--fee-bps", type=float, default=0.5)
    ap.add_argument("--slippage-bps", type=float, default=0.2)
    ap.add_argument("--alignment-mode", default="strict_asof", choices=["strict_asof", "legacy_index"])
    ap.add_argument("--alignment-version", default="strict_asof_v1")
    ap.add_argument("--max-feature-staleness-hours", type=int, default=24 * 14)
    ap.add_argument("--out-dir", default="artifacts/remediation_loops")
    ap.add_argument("--strict-targets", default="BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK")
    ap.add_argument("--signal-polarity-mode", default="auto_train_ic", choices=["normal", "auto_train_ic", "auto_train_pnl"])
    ap.add_argument("--auto-rebuild-backend-on-stale", action="store_true")
    ap.add_argument("--candidate-source", default="auto", choices=["none", "auto", "grid", "optuna", "file"])
    ap.add_argument("--candidate-top-k", type=int, default=8)
    ap.add_argument("--candidate-refresh-every", type=int, default=3)
    ap.add_argument("--candidate-file", default="")
    ap.add_argument("--candidate-grid-max-trials", type=int, default=24)
    ap.add_argument("--candidate-grid-timeout-sec", type=float, default=45.0)
    ap.add_argument("--candidate-grid-max-reject-rate", type=float, default=0.02)
    ap.add_argument("--candidate-optuna-log-glob", default="artifacts/hpo/optuna_trials_*.jsonl")
    ap.add_argument("--candidate-min-turnover", type=float, default=0.05)
    ap.add_argument("--candidate-min-trades", type=float, default=5.0)
    ap.add_argument("--candidate-min-abs-pnl", type=float, default=1e-5)
    ap.add_argument("--candidate-min-active-targets", type=int, default=2)
    ap.add_argument("--candidate-min-score", type=float, default=0.5)
    ap.add_argument("--fallback-entry-z", type=float, default=0.08)
    ap.add_argument("--fallback-exit-z", type=float, default=0.028)
    ap.add_argument("--fallback-base-weight", type=float, default=0.08)
    ap.add_argument("--fallback-high-vol-mult", type=float, default=0.35)
    ap.add_argument("--fallback-cost-lambda", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strict_targets = _parse_targets(args.strict_targets)

    strict_common = [
        "--include-sources",
        "prod",
        "--exclude-sources",
        "smoke,async_test,maintenance",
        "--data-regimes",
        "prod_live",
        "--score-source",
        "model",
    ]

    consecutive_green = 0
    iteration = 0
    started = datetime.now(timezone.utc)
    candidate_pool: List[Dict[str, Any]] = []
    candidate_last_refresh_iteration = 0

    while True:
        iteration += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        it_dir = out_dir / f"iter_{iteration:04d}_{ts}"
        it_dir.mkdir(parents=True, exist_ok=True)

        records: Dict[str, Any] = {
            "iteration": iteration,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "strict_filters": {
                "run_source": "prod",
                "score_source": "model",
                "data_regime": "prod_live",
            },
        }
        selected_candidate: Dict[str, Any] = {}
        if str(args.candidate_source).strip().lower() != "none":
            refresh_every = max(1, int(args.candidate_refresh_every))
            need_refresh = (
                not candidate_pool
                or candidate_last_refresh_iteration <= 0
                or ((iteration - candidate_last_refresh_iteration) >= refresh_every)
            )
            if need_refresh:
                discovery = _discover_candidates(args, strict_targets, it_dir=it_dir)
                records["candidate_discovery"] = discovery
                _write_json(it_dir / "candidate_discovery.json", discovery)
                candidate_pool = discovery.get("candidates") if isinstance(discovery.get("candidates"), list) else []
                if candidate_pool:
                    min_score = float(args.candidate_min_score)
                    filtered = [
                        c
                        for c in candidate_pool
                        if _safe_float(c.get("score"), default=-1e9) >= min_score
                    ]
                    records["candidate_filtering"] = {
                        "min_score": min_score,
                        "before": len(candidate_pool),
                        "after": len(filtered),
                    }
                    candidate_pool = filtered
                if candidate_pool:
                    candidate_last_refresh_iteration = iteration
                else:
                    fb = _fallback_candidate(args)
                    candidate_pool = [fb]
                    records["candidate_fallback_used"] = True
                    records["candidate_fallback"] = fb
            if candidate_pool:
                selected_candidate = candidate_pool[(iteration - 1) % len(candidate_pool)]
                records["selected_candidate"] = selected_candidate

        batch_cmd = [
            "python3",
            "scripts/run_prod_live_backtest_batch.py",
            "--api-base",
            args.api_base,
            "--n-runs",
            str(max(1, int(args.batch_runs))),
            "--targets",
            ",".join(strict_targets),
            "--lookback-days",
            "180",
            "--train-days",
            "56",
            "--test-days",
            "14",
            "--fee-bps",
            str(float(args.fee_bps)),
            "--slippage-bps",
            str(float(args.slippage_bps)),
            "--signal-polarity-mode",
            str(args.signal_polarity_mode),
            "--alignment-mode",
            str(args.alignment_mode),
            "--alignment-version",
            str(args.alignment_version),
            "--max-feature-staleness-hours",
            str(int(args.max_feature_staleness_hours)),
        ]
        if selected_candidate:
            batch_cmd.extend(_candidate_cli_args(selected_candidate))

        steps = [
            ("batch_backtest", batch_cmd),
            (
                "contract_validation",
                [
                    "python3",
                    "scripts/validate_backtest_contracts.py",
                    "--track",
                    "liquid",
                    "--lookback-days",
                    "180",
                    "--min-valid",
                    "20",
                    "--limit",
                    "500",
                    "--score-source",
                    "model",
                    "--include-sources",
                    "prod",
                    "--exclude-sources",
                    "smoke,async_test,maintenance",
                    "--data-regimes",
                    "prod_live",
                ],
            ),
            (
                "hard_metrics",
                [
                    "python3",
                    "scripts/evaluate_hard_metrics.py",
                    "--track",
                    "liquid",
                    "--lookback-days",
                    "180",
                    "--min-completed-runs",
                    "5",
                    "--min-observation-days",
                    "14",
                    *strict_common,
                ],
            ),
            (
                "no_leakage",
                [
                    "python3",
                    "scripts/validate_no_leakage.py",
                    "--track",
                    "liquid",
                    "--lookback-days",
                    "180",
                    "--score-source",
                    "model",
                    "--include-sources",
                    "prod",
                    "--exclude-sources",
                    "smoke,async_test,maintenance",
                    "--data-regimes",
                    "prod_live",
                ],
            ),
            (
                "parity",
                [
                    "python3",
                    "scripts/check_backtest_paper_parity.py",
                    "--track",
                    "liquid",
                    "--max-deviation",
                    "0.10",
                    "--min-completed-runs",
                    "5",
                    *strict_common,
                ],
            ),
            ("alerts", ["python3", "scripts/validate_phase45_alerts.py"]),
            ("readiness", ["python3", "scripts/check_gpu_cutover_readiness.py"]),
            ("snapshot", ["python3", "scripts/generate_status_snapshot.py", "--write"]),
        ]

        for name, cmd in steps:
            code, obj, out, err = _run_json(cmd)
            records[name] = {
                "cmd": cmd,
                "exit_code": code,
                "json": obj,
                "stdout": out,
                "stderr": err,
            }
            _write_json(it_dir / f"{name}.json", records[name])

        contracts_json = ((records.get("contract_validation") or {}).get("json") or {})
        stale_runtime = bool(contracts_json.get("suspected_backend_runtime_outdated"))
        rebuild_triggered = False
        if stale_runtime and bool(args.auto_rebuild_backend_on_stale):
            rebuild_triggered = True
            rb_code, rb_obj, rb_out, rb_err = _run_json(["docker", "compose", "build", "backend"])
            records["auto_rebuild_backend_build"] = {
                "cmd": ["docker", "compose", "build", "backend"],
                "exit_code": rb_code,
                "json": rb_obj,
                "stdout": rb_out,
                "stderr": rb_err,
            }
            _write_json(it_dir / "auto_rebuild_backend_build.json", records["auto_rebuild_backend_build"])
            up_code, up_obj, up_out, up_err = _run_json(["docker", "compose", "up", "-d", "backend"])
            records["auto_rebuild_backend_up"] = {
                "cmd": ["docker", "compose", "up", "-d", "backend"],
                "exit_code": up_code,
                "json": up_obj,
                "stdout": up_out,
                "stderr": up_err,
            }
            _write_json(it_dir / "auto_rebuild_backend_up.json", records["auto_rebuild_backend_up"])

        readiness = ((records.get("readiness") or {}).get("json") or {})
        ready = bool(readiness.get("ready_for_gpu_cutover"))
        if ready:
            consecutive_green += 1
        else:
            consecutive_green = 0

        summary = {
            "iteration": iteration,
            "started": records.get("started_at"),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "ready_for_gpu_cutover": ready,
            "consecutive_green": consecutive_green,
            "required_green_windows": int(args.green_windows),
            "gate_snapshot": (readiness or {}).get("gates") or {},
            "contract_passed": bool((((records.get("contract_validation") or {}).get("json") or {}).get("passed"))),
            "suspected_backend_runtime_outdated": stale_runtime,
            "auto_rebuild_backend_triggered": rebuild_triggered,
            "hard_status": str((((records.get("hard_metrics") or {}).get("json") or {}).get("status") or "unknown")),
            "no_leakage_passed": bool((((records.get("no_leakage") or {}).get("json") or {}).get("passed"))),
            "parity_status": str((((records.get("parity") or {}).get("json") or {}).get("status") or "unknown")),
            "candidate_source_mode": str(args.candidate_source),
            "candidate_pool_size": len(candidate_pool),
            "candidate_last_refresh_iteration": int(candidate_last_refresh_iteration),
            "selected_candidate": records.get("selected_candidate") or {},
        }
        _write_json(it_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False))

        if consecutive_green >= max(1, int(args.green_windows)):
            final = {
                "status": "ready",
                "message": "ready_for_gpu_cutover maintained across required windows",
                "iterations": iteration,
                "consecutive_green": consecutive_green,
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(out_dir / "final_ready.json", final)
            print(json.dumps(final, ensure_ascii=False))
            return 0

        if int(args.max_iterations) > 0 and iteration >= int(args.max_iterations):
            final = {
                "status": "not_ready",
                "message": "max iterations reached before readiness gates turned green",
                "iterations": iteration,
                "consecutive_green": consecutive_green,
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(out_dir / "final_not_ready.json", final)
            print(json.dumps(final, ensure_ascii=False))
            return 2

        if float(args.sleep_sec) > 0:
            time.sleep(float(args.sleep_sec))


if __name__ == "__main__":
    raise SystemExit(main())
