#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from _psql import run_psql


def _parse_sources(raw: str) -> List[str]:
    out: List[str] = []
    for part in (raw or "").split(","):
        s = part.strip().lower()
        if s and re.fullmatch(r"[a-z0-9_-]+", s):
            out.append(s)
    return out


def _parse_regimes(raw: str) -> List[str]:
    allowed = {"prod_live", "maintenance_replay", "mixed"}
    out: List[str] = []
    for part in (raw or "").split(","):
        s = part.strip().lower()
        if s in allowed:
            out.append(s)
    return out


def _sql_source_filters(include_sources: List[str], exclude_sources: List[str]) -> str:
    clauses: List[str] = []
    if include_sources:
        clauses.append("COALESCE(run_source,'prod') IN (" + ",".join(f"'{s}'" for s in include_sources) + ")")
    if exclude_sources:
        clauses.append("COALESCE(run_source,'prod') NOT IN (" + ",".join(f"'{s}'" for s in exclude_sources) + ")")
    return (" AND " + " AND ".join(clauses)) if clauses else ""


def _sql_regime_filter(data_regimes: List[str]) -> str:
    if not data_regimes:
        return ""
    return " AND COALESCE(NULLIF(config->>'data_regime',''),'missing') IN (" + ",".join(f"'{x}'" for x in data_regimes) + ")"


def _validate_metrics(metrics: Dict[str, Any]) -> List[str]:
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
    if "observation_days" in metrics:
        try:
            if int(metrics.get("observation_days") or 0) <= 0:
                missing.append("observation_days_positive")
        except Exception:
            missing.append("observation_days_positive")
    lck = (metrics or {}).get("leakage_checks") if isinstance(metrics, dict) else None
    if not isinstance(lck, dict):
        missing.append("leakage_checks_dict")
    else:
        if int(lck.get("leakage_violations", 0) or 0) > 0:
            missing.append("leakage_violations_zero")
        if not bool(lck.get("passed", False)):
            missing.append("leakage_checks_passed")
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate strict backtest metric contracts")
    ap.add_argument("--track", default="liquid")
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--score-source", default="model", choices=["model", "heuristic"])
    ap.add_argument("--include-sources", default="prod")
    ap.add_argument("--exclude-sources", default="smoke,async_test,maintenance")
    ap.add_argument("--data-regimes", default="prod_live")
    ap.add_argument("--include-superseded", action="store_true")
    ap.add_argument("--include-noncompleted", action="store_true")
    ap.add_argument("--enforce", action="store_true")
    ap.add_argument("--min-valid", type=int, default=20)
    args = ap.parse_args()

    include_sources = _parse_sources(args.include_sources)
    exclude_sources = _parse_sources(args.exclude_sources)
    data_regimes = _parse_regimes(args.data_regimes)

    supersede_filter = "" if args.include_superseded else " AND superseded_by_run_id IS NULL "
    completed_filter = "" if args.include_noncompleted else " AND COALESCE(metrics->>'status','')='completed' "

    sql = (
        "SELECT id::text, COALESCE(run_source,'prod'), "
        "COALESCE(config->>'score_source','heuristic'), "
        "COALESCE(config->>'data_regime','missing'), "
        "COALESCE(metrics::text,'{}') "
        "FROM backtest_runs "
        f"WHERE track='{str(args.track).strip().lower()}' "
        f"AND created_at > NOW() - make_interval(days => {int(args.lookback_days)}) "
        + _sql_source_filters(include_sources, exclude_sources)
        + _sql_regime_filter(data_regimes)
        + supersede_filter
        + completed_filter
        + f" AND COALESCE(config->>'score_source','heuristic')='{args.score_source}' "
        "ORDER BY created_at DESC "
        f"LIMIT {max(1, int(args.limit))};"
    )

    rows = [r for r in run_psql(sql).splitlines() if r.strip()]
    sql_shadow = (
        "SELECT COUNT(*)::text FROM backtest_runs "
        f"WHERE track='{str(args.track).strip().lower()}' "
        f"AND created_at > NOW() - make_interval(days => {int(args.lookback_days)}) "
        "AND COALESCE(run_source,'prod')='prod' "
        + supersede_filter
        + f"AND COALESCE(config->>'score_source','heuristic')='{args.score_source}' "
        "AND COALESCE(NULLIF(config->>'data_regime',''),'missing')='missing';"
    )
    raw_shadow = run_psql(sql_shadow)
    prod_model_missing_regime_rows = int(float(raw_shadow or 0)) if raw_shadow else 0
    valid = 0
    invalid = 0
    issues: List[Dict[str, Any]] = []
    regime_dist: Dict[str, int] = {}
    for line in rows:
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        run_id_s, run_source, score_source, data_regime, metrics_raw = parts
        regime_dist[data_regime] = regime_dist.get(data_regime, 0) + 1
        try:
            metrics = json.loads(metrics_raw)
        except Exception:
            metrics = {}
        missing = _validate_metrics(metrics)
        if missing:
            invalid += 1
            issues.append(
                {
                    "run_id": int(float(run_id_s or 0)),
                    "run_source": run_source,
                    "score_source": score_source,
                    "data_regime": data_regime,
                    "missing": missing,
                }
            )
        else:
            valid += 1

    out = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "window_start": (datetime.now(timezone.utc) - timedelta(days=int(args.lookback_days))).isoformat(),
        "window_end": datetime.now(timezone.utc).isoformat(),
        "track": str(args.track).strip().lower(),
        "score_source": args.score_source,
        "include_sources": include_sources,
        "exclude_sources": exclude_sources,
        "data_regimes": data_regimes,
        "rows_total": len(rows),
        "valid_contract_rows": valid,
        "invalid_contract_rows": invalid,
        "min_valid_required": int(args.min_valid),
        "regime_distribution": regime_dist,
        "prod_model_missing_regime_rows": int(prod_model_missing_regime_rows),
        "suspected_backend_runtime_outdated": bool(len(rows) == 0 and prod_model_missing_regime_rows > 0),
        "passed": invalid == 0 and valid >= int(args.min_valid),
        "issues": issues[:50],
    }
    print(json.dumps(out, ensure_ascii=False))

    if args.enforce and (not out["passed"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
