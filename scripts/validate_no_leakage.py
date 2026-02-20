#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from _psql import run_psql


def _to_bool(raw: Any) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_read_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _safe_read_last_jsonl(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    last = ""
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line.strip()
    except Exception:
        return {}
    if not last:
        return {}
    try:
        obj = json.loads(last)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _parse_utc_timestamp(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _evaluate_candidate_gate_check(
    *,
    registry_path: str,
    require_passed: bool,
    max_age_hours: float,
    now: datetime,
) -> Dict[str, Any]:
    row = _safe_read_last_jsonl(registry_path)
    gate = row.get("gate") if isinstance(row.get("gate"), dict) else {}
    passed = bool(gate.get("passed", False))
    reasons = [str(x) for x in list(gate.get("reasons") or []) if str(x).strip()]
    ts = _parse_utc_timestamp(row.get("registered_at"))
    age_hours = None
    stale = False
    if ts is not None:
        age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
        stale = bool(max_age_hours > 0 and age_hours > max_age_hours)
    exists = bool(row)
    status = "ok" if exists else "missing"
    effective_passed = True
    effective_reasons: List[str] = []
    if require_passed:
        if not exists:
            effective_passed = False
            effective_reasons.append("candidate_registry_missing")
        elif not passed:
            effective_passed = False
            effective_reasons.append("candidate_gate_not_passed")
        if stale:
            effective_passed = False
            effective_reasons.append("candidate_registry_stale")
    return {
        "required": bool(require_passed),
        "status": status,
        "registry_path": str(registry_path),
        "exists": exists,
        "gate_passed": bool(passed),
        "gate_reasons": reasons,
        "registered_at": row.get("registered_at"),
        "age_hours": age_hours,
        "stale": stale,
        "effective_passed": bool(effective_passed),
        "effective_reasons": effective_reasons,
    }


def _evaluate_multimodal_gate_snapshot_check(
    *,
    snapshot_path: str,
    track: str,
    require_passed: bool,
    max_age_hours: float,
    now: datetime,
) -> Dict[str, Any]:
    snap = _safe_read_json(snapshot_path)
    exists = bool(snap)
    passed = bool(snap.get("passed", False))
    reasons = [str(x) for x in list(snap.get("reasons") or []) if str(x).strip()]
    ts = _parse_utc_timestamp(snap.get("evaluated_at"))
    age_hours = None
    stale = False
    if ts is not None:
        age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
        stale = bool(max_age_hours > 0 and age_hours > max_age_hours)
    gate_track = str(snap.get("track") or "").strip().lower()
    expected_track = str(track or "").strip().lower()
    track_match = bool((not gate_track) or (not expected_track) or gate_track == expected_track)
    status = "ok" if exists else "missing"
    effective_passed = True
    effective_reasons: List[str] = []
    if require_passed:
        if not exists:
            effective_passed = False
            effective_reasons.append("multimodal_gate_snapshot_missing")
        elif not passed:
            effective_passed = False
            effective_reasons.append("multimodal_gate_not_passed")
        if not track_match:
            effective_passed = False
            effective_reasons.append("multimodal_gate_track_mismatch")
        if stale:
            effective_passed = False
            effective_reasons.append("multimodal_gate_snapshot_stale")
    return {
        "required": bool(require_passed),
        "status": status,
        "snapshot_path": str(snapshot_path),
        "exists": exists,
        "track": gate_track,
        "track_match": track_match,
        "passed": bool(passed),
        "reasons": reasons,
        "evaluated_at": snap.get("evaluated_at"),
        "age_hours": age_hours,
        "stale": stale,
        "effective_passed": bool(effective_passed),
        "effective_reasons": effective_reasons,
    }


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
    ap.add_argument("--candidate-registry", default=os.getenv("CANDIDATE_REGISTRY_FILE", "artifacts/models/candidate_registry.jsonl"))
    ap.add_argument("--require-candidate-gate-passed", default=os.getenv("VALIDATE_REQUIRE_CANDIDATE_GATE_PASSED", "0"))
    ap.add_argument("--candidate-max-age-hours", type=float, default=float(os.getenv("VALIDATE_CANDIDATE_MAX_AGE_HOURS", "0")))
    ap.add_argument("--multimodal-gate-snapshot", default=os.getenv("MULTIMODAL_GATE_SNAPSHOT", "artifacts/ops/multimodal_gate_state.json"))
    ap.add_argument("--require-multimodal-gate-passed", default=os.getenv("VALIDATE_REQUIRE_MULTIMODAL_GATE_PASSED", "0"))
    ap.add_argument(
        "--multimodal-gate-max-age-hours",
        type=float,
        default=float(os.getenv("VALIDATE_MULTIMODAL_GATE_MAX_AGE_HOURS", "0")),
    )
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
    rows = [r for r in run_psql(sql).splitlines() if r.strip()]
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
    leakage_passed = bool(checked > 0 and violations == 0)
    candidate_gate_check = _evaluate_candidate_gate_check(
        registry_path=str(args.candidate_registry),
        require_passed=_to_bool(args.require_candidate_gate_passed),
        max_age_hours=float(args.candidate_max_age_hours),
        now=now,
    )
    multimodal_gate_check = _evaluate_multimodal_gate_snapshot_check(
        snapshot_path=str(args.multimodal_gate_snapshot),
        track=str(args.track).strip().lower(),
        require_passed=_to_bool(args.require_multimodal_gate_passed),
        max_age_hours=float(args.multimodal_gate_max_age_hours),
        now=now,
    )
    gates = {
        "leakage_passed": bool(leakage_passed),
        "candidate_gate_passed": bool(candidate_gate_check.get("effective_passed", True)),
        "multimodal_gate_passed": bool(multimodal_gate_check.get("effective_passed", True)),
    }
    out = {
        "evaluated_at": now.isoformat(),
        "window_start": (now - timedelta(days=max(1, int(args.lookback_days)))).isoformat(),
        "window_end": now.isoformat(),
        "track": str(args.track).strip().lower(),
        "checked_runs": int(checked),
        "violating_runs": int(violations),
        "fallback_alignment_runs": int(fallback_modes),
        "candidate_gate_check": candidate_gate_check,
        "multimodal_gate_check": multimodal_gate_check,
        "gates": gates,
        "passed": bool(all(bool(v) for v in gates.values())),
        "issues": issues[:50],
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
