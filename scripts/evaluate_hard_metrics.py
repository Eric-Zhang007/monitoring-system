#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any, Dict, List

from _metrics_test_logger import record_metrics_test


def _sharpe(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    sd = pstdev(vals)
    if sd <= 1e-12:
        return 0.0
    return float((mu / sd) * math.sqrt(252.0))


def _sharpe_with_horizon(vals: List[float], sample_days: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    sd = pstdev(vals)
    if sd <= 1e-12:
        return 0.0
    avg_days = mean(sample_days) if sample_days else 1.0
    avg_days = max(1.0, float(avg_days))
    annual_factor = math.sqrt(365.0 / avg_days)
    return float((mu / sd) * annual_factor)


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


def _run_psql(sql: str) -> str:
    cmd = [
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
        sql,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def _parse_sources(raw: str) -> List[str]:
    out: List[str] = []
    for part in (raw or "").split(","):
        s = part.strip().lower()
        if not s:
            continue
        if not re.fullmatch(r"[a-z0-9_-]+", s):
            continue
        out.append(s)
    return out


def _parse_regimes(raw: str) -> List[str]:
    allowed = {"prod_live", "maintenance_replay", "mixed"}
    out: List[str] = []
    for part in (raw or "").split(","):
        s = part.strip().lower()
        if not s or s not in allowed:
            continue
        out.append(s)
    return out


def _sql_source_filters(include_sources: List[str], exclude_sources: List[str]) -> str:
    clauses: List[str] = []
    if include_sources:
        inc = ",".join(f"'{s}'" for s in include_sources)
        clauses.append(f"COALESCE(run_source,'prod') IN ({inc})")
    if exclude_sources:
        exc = ",".join(f"'{s}'" for s in exclude_sources)
        clauses.append(f"COALESCE(run_source,'prod') NOT IN ({exc})")
    return (" AND " + " AND ".join(clauses)) if clauses else ""


def _sql_regime_filters(data_regimes: List[str]) -> str:
    if not data_regimes:
        return ""
    vals = ",".join(f"'{r}'" for r in data_regimes)
    return f" AND COALESCE(NULLIF(config->>'data_regime',''),'missing') IN ({vals}) "


def _sql_target_filter(targets: List[str]) -> str:
    if not targets:
        return ""
    conds = []
    for t in targets:
        conds.append(
            "EXISTS (SELECT 1 FROM jsonb_array_elements_text(COALESCE(config->'targets','[]'::jsonb)) x(v) "
            f"WHERE UPPER(v)=UPPER('{t}'))"
        )
    return " AND (" + " OR ".join(conds) + ")"


def main() -> int:
    started_at = datetime.now(timezone.utc)
    ap = argparse.ArgumentParser(description="Evaluate codex-plan hard metrics")
    ap.add_argument("--track", default="liquid")
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--score-source", default="model", choices=["model", "heuristic"])
    ap.add_argument("--enforce", action="store_true")
    ap.add_argument("--include-sources", default=os.getenv("BACKTEST_GATE_INCLUDE_SOURCES", "prod"))
    ap.add_argument("--exclude-sources", default=os.getenv("BACKTEST_GATE_EXCLUDE_SOURCES", "smoke,async_test,maintenance"))
    ap.add_argument("--data-regimes", default=os.getenv("BACKTEST_GATE_DATA_REGIMES", "prod_live"))
    ap.add_argument("--targets", default="", help="comma-separated targets filter")
    ap.add_argument("--include-superseded", action="store_true")
    ap.add_argument("--min-completed-runs", type=int, default=5)
    ap.add_argument("--min-observation-days", type=int, default=int(os.getenv("BACKTEST_GATE_MIN_OBSERVATION_DAYS", "14")))
    ap.add_argument("--min-sharpe-daily", type=float, default=float(os.getenv("BACKTEST_GATE_MIN_SHARPE_DAILY", "1.5")))
    args = ap.parse_args()

    track = str(args.track).strip().lower()
    score_source = str(args.score_source).strip().lower()
    track_mode = "liquid_strict" if track == "liquid" else "vc_monitor"
    monitor_only = track != "liquid"

    include_sources = _parse_sources(args.include_sources)
    exclude_sources = _parse_sources(args.exclude_sources)
    data_regimes = _parse_regimes(args.data_regimes)
    source_cond = _sql_source_filters(include_sources, exclude_sources)
    regime_cond = _sql_regime_filters(data_regimes)
    supersede_cond = "" if args.include_superseded else " AND superseded_by_run_id IS NULL "
    targets = [s.strip().upper() for s in str(args.targets).split(",") if s.strip()]
    target_cond = _sql_target_filter(targets)
    required_contract_keys = [
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

    sql_runs = (
        "SELECT "
        "COALESCE(metrics->>'status',''), "
        "COALESCE(metrics->>'reason',''), "
        "COALESCE(metrics->>'pnl_after_cost','0'), "
        "COALESCE(metrics->>'max_drawdown','0'), "
        "COALESCE(metrics->>'sharpe_daily', metrics->>'sharpe', '0'), "
        "COALESCE(metrics->>'observation_days','0'), "
        "COALESCE(metrics->'leakage_checks'->>'passed','false'), "
        "COALESCE(metrics->'leakage_checks'->>'leakage_violations','0'), "
        "CASE WHEN superseded_by_run_id IS NULL THEN '0' ELSE '1' END, "
        "COALESCE(config->>'lookback_days','0'), "
        "COALESCE(config->>'data_regime','missing') "
        "FROM backtest_runs "
        f"WHERE track='{track}' "
        f"AND created_at > NOW() - make_interval(days => {args.lookback_days}) "
        f"{source_cond} "
        f"{regime_cond} "
        f"{supersede_cond} "
        f"{target_cond} "
        f"AND COALESCE(config->>'score_source','heuristic') = '{score_source}' "
        "AND metrics ? 'pnl_after_cost' "
        "AND metrics ? 'max_drawdown' "
        "ORDER BY created_at DESC LIMIT 1000;"
    )
    rows = [r for r in _run_psql(sql_runs).splitlines() if r.strip()]
    missing_contract_expr = " OR ".join([f"NOT (metrics ? '{k}')" for k in required_contract_keys])
    sql_contract = (
        "SELECT COUNT(*)::text "
        "FROM backtest_runs "
        f"WHERE track='{track}' "
        f"AND created_at > NOW() - make_interval(days => {args.lookback_days}) "
        f"{source_cond} "
        f"{regime_cond} "
        f"{supersede_cond} "
        f"{target_cond} "
        f"AND COALESCE(config->>'score_source','heuristic') = '{score_source}' "
        "AND COALESCE(metrics->>'status','')='completed' "
        f"AND ({missing_contract_expr});"
    )
    raw_missing_contract_count = _run_psql(sql_contract)
    missing_required_metrics_count = int(float(raw_missing_contract_count or 0)) if raw_missing_contract_count else 0

    completed_pnls: List[float] = []
    completed_dds: List[float] = []
    completed_sharpe_daily: List[float] = []
    completed_observation_days: List[float] = []
    completed_horizon_days: List[float] = []
    failed_runs_count = 0
    artifact_failures = 0
    failed_effective_count = 0
    artifact_failures_effective = 0
    leakage_fail_runs = 0
    superseded_runs_count = 0
    total_runs = len(rows)
    effective_total_runs = 0
    regime_counts: Dict[str, int] = {}
    for row in rows:
        parts = row.split("|")
        if len(parts) != 11:
            continue
        (
            status,
            reason,
            pnl_raw,
            dd_raw,
            sharpe_daily_raw,
            obs_days_raw,
            leak_pass_raw,
            leak_viol_raw,
            superseded_flag,
            lookback_raw,
            regime_raw,
        ) = parts
        status = (status or "").strip().lower()
        reason = (reason or "").strip().lower()
        regime = (regime_raw or "missing").strip().lower() or "missing"
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        superseded = str(superseded_flag or "0").strip() == "1"
        if superseded:
            superseded_runs_count += 1
            continue
        effective_total_runs += 1
        if status == "completed":
            try:
                leak_pass = str(leak_pass_raw or "").strip().lower() in {"1", "t", "true", "yes", "y"}
                leak_viol = int(float(leak_viol_raw or 0))
                if (not leak_pass) or leak_viol > 0:
                    leakage_fail_runs += 1
                completed_pnls.append(float(pnl_raw))
                completed_dds.append(float(dd_raw))
                completed_sharpe_daily.append(float(sharpe_daily_raw or 0.0))
                completed_observation_days.append(float(obs_days_raw or 0.0))
                completed_horizon_days.append(max(1.0, float(lookback_raw or 0.0)))
            except ValueError:
                continue
        else:
            failed_runs_count += 1
            if reason == "model_artifact_missing":
                artifact_failures += 1
            if not superseded:
                failed_effective_count += 1
                if reason == "model_artifact_missing":
                    artifact_failures_effective += 1

    sql_exec = (
        "SELECT COUNT(*) FILTER (WHERE status='rejected')::text || ',' || COUNT(*)::text "
        "FROM orders_sim "
        f"WHERE track='{track}' "
        + (f" AND UPPER(target) IN ({','.join([repr(t) for t in targets])}) " if targets else "")
        + " "
        f"AND created_at > NOW() - make_interval(days => {args.lookback_days});"
    )
    out_exec = _run_psql(sql_exec)
    rejected = 0
    total_orders = 0
    if out_exec and "," in out_exec:
        a, b = out_exec.split(",", 1)
        rejected = int(float(a or 0))
        total_orders = int(float(b or 0))

    sharpe_daily = float(mean(completed_sharpe_daily)) if completed_sharpe_daily else 0.0
    maxdd = max(completed_dds) if completed_dds else 0.0
    observation_days = int(min(completed_observation_days)) if completed_observation_days else 0
    reject_rate = (rejected / total_orders) if total_orders > 0 else 0.0
    failed_ratio = (failed_effective_count / effective_total_runs) if effective_total_runs > 0 else 0.0
    artifact_failure_ratio = (
        artifact_failures_effective / max(1, failed_effective_count)
        if failed_effective_count > 0
        else 0.0
    )

    insufficient_runs = len(completed_pnls) < int(args.min_completed_runs)
    insufficient_observation = observation_days < int(args.min_observation_days)
    insufficient = insufficient_runs or insufficient_observation
    checks: Dict[str, Any] = {
        "min_completed_runs_ok": not insufficient_runs,
        "observation_days_ok": not insufficient_observation,
        "oos_sharpe_daily_gt_threshold": sharpe_daily > float(args.min_sharpe_daily) if not insufficient else False,
        "legacy_oos_sharpe_daily_gt_1_5": sharpe_daily > 1.5 if not insufficient else False,
        "maxdd_lt_0_12": maxdd < 0.12 if not insufficient else False,
        "reject_rate_lt_0_01": reject_rate < 0.01,
        "leakage_checks_passed_all": leakage_fail_runs == 0,
    }
    gate_check_keys = [
        "min_completed_runs_ok",
        "observation_days_ok",
        "oos_sharpe_daily_gt_threshold",
        "maxdd_lt_0_12",
        "reject_rate_lt_0_01",
        "leakage_checks_passed_all",
    ]
    hard_passed = all(bool(checks.get(k, False)) for k in gate_check_keys) and (not insufficient)
    passed = True if monitor_only else hard_passed

    out: Dict[str, Any] = {
        "track": track,
        "score_source": score_source,
        "track_mode": track_mode,
        "monitor_only": monitor_only,
        "evaluated_at": started_at.isoformat(),
        "window_start": (started_at - timedelta(days=int(args.lookback_days))).isoformat(),
        "window_end": started_at.isoformat(),
        "lookback_days": args.lookback_days,
        "min_completed_runs": int(args.min_completed_runs),
        "include_sources": include_sources,
        "exclude_sources": exclude_sources,
        "data_regimes": data_regimes,
        "data_regime_distribution": regime_counts,
        "targets_filter": targets,
        "samples_completed": len(completed_pnls),
        "samples_total": total_runs,
        "samples_effective_total": effective_total_runs,
        "missing_required_metrics_count": int(missing_required_metrics_count),
        "failed_runs_count": failed_runs_count,
        "failed_runs_effective_count": failed_effective_count,
        "artifact_missing_effective_count": artifact_failures_effective,
        "leakage_fail_runs": int(leakage_fail_runs),
        "superseded_runs_count": superseded_runs_count,
        "status": "insufficient_observation" if insufficient else ("passed" if hard_passed else "failed"),
        "failed_ratio": round(failed_ratio, 6),
        "artifact_failure_ratio": round(artifact_failure_ratio, 6),
        "sharpe": round(sharpe_daily, 6),
        "sharpe_daily": round(sharpe_daily, 6),
        "sharpe_method": "daily_agg_v1",
        "observation_days": int(observation_days),
        "min_observation_days": int(args.min_observation_days),
        "min_sharpe_daily": float(args.min_sharpe_daily),
        "max_drawdown": round(maxdd, 6),
        "execution_reject_rate": round(reject_rate, 6),
        "pnl_direction_adjusted": True,
        "checks": checks,
        "hard_passed": hard_passed,
        "passed": passed,
    }
    print(json.dumps(out, ensure_ascii=False))
    record_metrics_test(
        test_name="evaluate_hard_metrics",
        payload=out,
        window_start=str(out.get("window_start")),
        window_end=str(out.get("window_end")),
        extra={
            "argv": {
                "track": track,
                "score_source": score_source,
                "lookback_days": int(args.lookback_days),
                "enforce": bool(args.enforce),
                "min_completed_runs": int(args.min_completed_runs),
                "min_observation_days": int(args.min_observation_days),
                "include_sources": include_sources,
                "exclude_sources": exclude_sources,
                "data_regimes": data_regimes,
                "targets": targets,
            }
        },
    )

    if args.enforce and (not monitor_only) and (not hard_passed):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
