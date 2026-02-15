#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


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
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "psql failed")
    return p.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate strict no-leakage backtest evidence")
    ap.add_argument("--track", default="liquid")
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--include-sources", default="prod")
    ap.add_argument("--exclude-sources", default="smoke,async_test,maintenance")
    ap.add_argument("--score-source", default="model", choices=["model", "heuristic"])
    ap.add_argument("--data-regimes", default="prod_live")
    ap.add_argument("--include-superseded", action="store_true")
    ap.add_argument("--include-noncompleted", action="store_true")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()

    include = [s.strip().lower() for s in str(args.include_sources).split(",") if s.strip()]
    exclude = [s.strip().lower() for s in str(args.exclude_sources).split(",") if s.strip()]
    regimes = [s.strip().lower() for s in str(args.data_regimes).split(",") if s.strip()]
    src_inc = f"AND COALESCE(run_source,'prod') IN ({','.join(repr(s) for s in include)})" if include else ""
    src_exc = f"AND COALESCE(run_source,'prod') NOT IN ({','.join(repr(s) for s in exclude)})" if exclude else ""
    reg = f"AND COALESCE(NULLIF(config->>'data_regime',''),'missing') IN ({','.join(repr(s) for s in regimes)})" if regimes else ""
    supersede_filter = "" if args.include_superseded else "AND superseded_by_run_id IS NULL"
    completed_filter = "" if args.include_noncompleted else "AND COALESCE(metrics->>'status','')='completed'"

    sql = f"""
    SELECT id::text, COALESCE(metrics::text, '{{}}')
    FROM backtest_runs
    WHERE track='{str(args.track).strip().lower()}'
      AND created_at > NOW() - make_interval(days => {max(1, int(args.lookback_days))})
      AND COALESCE(config->>'score_source','heuristic')='{str(args.score_source).strip().lower()}'
      {src_inc}
      {src_exc}
      {reg}
      {supersede_filter}
      {completed_filter}
    ORDER BY created_at DESC
    LIMIT {max(1, int(args.limit))};
    """
    rows = [r for r in _run_psql(sql).splitlines() if r.strip()]
    checked = 0
    violations = 0
    fallback_modes = 0
    issues: List[Dict[str, Any]] = []
    for line in rows:
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue
        run_id_s, metrics_raw = parts
        try:
            metrics = json.loads(metrics_raw)
        except Exception:
            metrics = {}
        checked += 1
        leak = (metrics.get("leakage_checks") or {}) if isinstance(metrics, dict) else {}
        audit = (metrics.get("alignment_audit") or {}) if isinstance(metrics, dict) else {}
        leak_viol = int(leak.get("leakage_violations", 0) or 0)
        leak_pass = bool(leak.get("passed", False))
        mode = str(audit.get("alignment_mode_applied") or "")
        if "fallback" in mode.lower():
            fallback_modes += 1
        if leak_viol > 0 or (not leak_pass):
            violations += 1
            issues.append(
                {
                    "run_id": int(float(run_id_s or 0)),
                    "leakage_violations": leak_viol,
                    "leakage_passed": leak_pass,
                    "alignment_mode_applied": mode,
                }
            )

    now = datetime.now(timezone.utc)
    out = {
        "evaluated_at": now.isoformat(),
        "window_start": (now - timedelta(days=max(1, int(args.lookback_days)))).isoformat(),
        "window_end": now.isoformat(),
        "track": str(args.track).strip().lower(),
        "checked_runs": int(checked),
        "violating_runs": int(violations),
        "fallback_alignment_runs": int(fallback_modes),
        "passed": bool(checked > 0 and violations == 0),
        "issues": issues[:50],
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
