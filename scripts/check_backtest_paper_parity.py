#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from _metrics_test_logger import record_metrics_test


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


def _parse_sources(raw: str) -> list[str]:
    out: list[str] = []
    for part in (raw or "").split(","):
        s = part.strip().lower()
        if not s:
            continue
        if not re.fullmatch(r"[a-z0-9_-]+", s):
            continue
        out.append(s)
    return out


def _parse_regimes(raw: str) -> list[str]:
    allowed = {"prod_live", "maintenance_replay", "mixed"}
    out: list[str] = []
    for part in (raw or "").split(","):
        s = part.strip().lower()
        if not s or s not in allowed:
            continue
        out.append(s)
    return out


def _sql_source_filters(include_sources: list[str], exclude_sources: list[str]) -> str:
    clauses: list[str] = []
    if include_sources:
        inc = ",".join(f"'{s}'" for s in include_sources)
        clauses.append(f"COALESCE(run_source,'prod') IN ({inc})")
    if exclude_sources:
        exc = ",".join(f"'{s}'" for s in exclude_sources)
        clauses.append(f"COALESCE(run_source,'prod') NOT IN ({exc})")
    return (" AND " + " AND ".join(clauses)) if clauses else ""


def _sql_regime_filters(data_regimes: list[str]) -> str:
    if not data_regimes:
        return ""
    vals = ",".join(f"'{r}'" for r in data_regimes)
    return f" AND COALESCE(NULLIF(config->>'data_regime',''),'missing') IN ({vals}) "


def _sql_targets_filter(targets: list[str]) -> str:
    if not targets:
        return ""
    conds = []
    for t in targets:
        conds.append(
            "EXISTS (SELECT 1 FROM jsonb_array_elements_text(COALESCE(config->'targets','[]'::jsonb)) x(v) "
            f"WHERE UPPER(v)=UPPER('{t}'))"
        )
    return " AND (" + " OR ".join(conds) + ")"


def _bt_target_map(
    track: str,
    hours: int,
    include_sources: list[str],
    exclude_sources: list[str],
    targets_filter: list[str],
    score_source: str,
    data_regimes: list[str],
) -> dict[str, dict[str, float]]:
    source_cond = _sql_source_filters(include_sources, exclude_sources)
    regime_cond = _sql_regime_filters(data_regimes)
    target_cond = _sql_targets_filter(targets_filter)
    sql = (
        "SELECT config::text, metrics::text "
        "FROM backtest_runs "
        f"WHERE track='{track}' "
        "AND created_at > NOW() - make_interval(hours => " + str(hours) + ") "
        "AND COALESCE(metrics->>'status','')='completed' "
        f"AND COALESCE(config->>'score_source','heuristic')='{score_source}' "
        + source_cond + " "
        + regime_cond + " "
        + target_cond + " "
        "AND superseded_by_run_id IS NULL;"
    )
    rows = [r for r in _run_psql(sql).splitlines() if r.strip()]
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        parts = row.split("|", 1)
        if len(parts) != 2:
            continue
        try:
            cfg = json.loads(parts[0])
            metrics = json.loads(parts[1])
        except Exception:
            continue
        targets = cfg.get("targets") if isinstance(cfg.get("targets"), list) else []
        if not targets:
            continue
        per_target = metrics.get("per_target") if isinstance(metrics.get("per_target"), dict) else {}
        for t in targets:
            target = str(t).upper()
            if target not in out:
                out[target] = {"sum": 0.0, "count": 0.0}
            if isinstance(per_target.get(target), dict):
                pnl = float((per_target.get(target) or {}).get("pnl_after_cost", 0.0) or 0.0)
            elif target in per_target:
                pnl = float(per_target.get(target) or 0.0)
            else:
                pnl = float(metrics.get("pnl_after_cost", 0.0) or 0.0) / max(1.0, float(len(targets)))
            out[target]["sum"] += pnl
            out[target]["count"] += 1.0
    return out


def _paper_target_map(track: str, hours: int, targets_filter: list[str]) -> dict[str, dict[str, float]]:
    target_cond = ""
    if targets_filter:
        target_cond = " AND UPPER(target) IN (" + ",".join(f"'{t}'" for t in targets_filter) + ") "
    sql = (
        "SELECT UPPER(target), side, "
        "COALESCE((metadata->'execution'->>'filled_qty')::double precision, quantity::double precision, 0.0), "
        "COALESCE((metadata->'execution'->>'avg_fill_price')::double precision, est_price::double precision, 0.0), "
        "COALESCE(est_price::double precision, 0.0) "
        "FROM orders_sim "
        f"WHERE track='{track}' "
        "AND created_at > NOW() - make_interval(hours => " + str(hours) + ") "
        + target_cond +
        "AND status IN ('filled','partially_filled');"
    )
    rows = [r for r in _run_psql(sql).splitlines() if r.strip()]
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        parts = row.split("|")
        if len(parts) != 5:
            continue
        target, side, qty_raw, fill_raw, est_raw = parts
        est = float(est_raw or 0.0)
        qty = float(qty_raw or 0.0)
        fill = float(fill_raw or 0.0)
        if est <= 0 or qty <= 0:
            continue
        realized = (fill - est) / max(abs(est), 1e-12)
        if side.lower() == "sell":
            realized = -realized
        notional = abs(qty * est)
        if target not in out:
            out[target] = {"sum_weighted": 0.0, "sum_notional": 0.0, "orders": 0.0}
        out[target]["sum_weighted"] += realized * notional
        out[target]["sum_notional"] += notional
        out[target]["orders"] += 1.0
    return out


def _window_pair(
    bt_map: dict[str, dict[str, float]], px_map: dict[str, dict[str, float]], matched: list[str], parity_floor: float
) -> tuple[float, float, float, int]:
    bt_sum = 0.0
    px_sum = 0.0
    w_sum = 0.0
    orders = 0
    for t in matched:
        bt_avg = float((bt_map.get(t) or {}).get("sum", 0.0)) / max(1.0, float((bt_map.get(t) or {}).get("count", 0.0)))
        px_notional = float((px_map.get(t) or {}).get("sum_notional", 0.0))
        px_avg = float((px_map.get(t) or {}).get("sum_weighted", 0.0)) / max(1e-12, px_notional)
        w = max(1.0, px_notional)
        bt_sum += bt_avg * w
        px_sum += px_avg * w
        w_sum += w
        orders += int((px_map.get(t) or {}).get("orders", 0.0))
    if w_sum <= 0:
        return 0.0, 0.0, 0.0, orders
    bt_avg = bt_sum / w_sum
    px_avg = px_sum / w_sum
    delta = abs(bt_avg - px_avg)
    rel = delta / max(1e-6, parity_floor, abs(bt_avg), abs(px_avg))
    return bt_avg, px_avg, rel, orders


def main() -> int:
    started_at = datetime.now(timezone.utc)
    p = argparse.ArgumentParser(description="Check backtest vs paper PnL parity")
    p.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    p.add_argument("--track", default="liquid")
    p.add_argument("--max-deviation", type=float, default=0.10, help="max allowed relative deviation")
    p.add_argument("--score-source", default="model", choices=["model", "heuristic"])
    p.add_argument("--min-completed-runs", type=int, default=5)
    p.add_argument("--parity-floor", type=float, default=float(os.getenv("PARITY_RETURN_FLOOR", "0.02")))
    p.add_argument("--include-sources", default=os.getenv("BACKTEST_GATE_INCLUDE_SOURCES", "prod"))
    p.add_argument("--exclude-sources", default=os.getenv("BACKTEST_GATE_EXCLUDE_SOURCES", "smoke,async_test,maintenance"))
    p.add_argument("--data-regimes", default=os.getenv("BACKTEST_GATE_DATA_REGIMES", "prod_live"))
    p.add_argument("--targets", default="", help="comma-separated targets filter")
    args = p.parse_args()
    include_sources = _parse_sources(args.include_sources)
    exclude_sources = _parse_sources(args.exclude_sources)
    data_regimes = _parse_regimes(args.data_regimes)
    score_source = str(args.score_source).strip().lower()
    targets_filter = [s.strip().upper() for s in str(args.targets).split(",") if s.strip()]
    source_cond = _sql_source_filters(include_sources, exclude_sources)
    regime_cond = _sql_regime_filters(data_regimes)
    target_cond = _sql_targets_filter(targets_filter)

    sql = (
        "SELECT COUNT(*)::text "
        "FROM backtest_runs "
        f"WHERE track='{args.track}' "
        "AND COALESCE(metrics->>'status','')='completed' "
        f"AND COALESCE(config->>'score_source','heuristic')='{score_source}' "
        + source_cond + " "
        + regime_cond + " "
        + target_cond + " "
        "AND superseded_by_run_id IS NULL "
        "AND created_at > NOW() - make_interval(days => 30);"
    )
    raw = _run_psql(sql)
    completed_runs = int(float(raw or 0)) if raw else 0

    if completed_runs < args.min_completed_runs:
        out = {
            "status": "insufficient_observation",
            "passed": False,
            "evaluated_at": started_at.isoformat(),
            "window_start": (started_at - timedelta(days=30)).isoformat(),
            "window_end": started_at.isoformat(),
            "completed_runs": completed_runs,
            "score_source": score_source,
            "min_completed_runs": args.min_completed_runs,
            "reason": "insufficient_completed_backtests",
        }
        print(json.dumps(out, ensure_ascii=False))
        record_metrics_test(
            test_name="check_backtest_paper_parity",
            payload=out,
            window_start=str(out.get("window_start")),
            window_end=str(out.get("window_end")),
            extra={
                "argv": {
                    "track": args.track,
                    "max_deviation": args.max_deviation,
                    "min_completed_runs": args.min_completed_runs,
                    "parity_floor": args.parity_floor,
                    "include_sources": include_sources,
                    "exclude_sources": exclude_sources,
                    "data_regimes": data_regimes,
                }
            },
        )
        return 0

    bt_7 = _bt_target_map(args.track, 24 * 7, include_sources, exclude_sources, targets_filter, score_source, data_regimes)
    bt_30 = _bt_target_map(args.track, 24 * 30, include_sources, exclude_sources, targets_filter, score_source, data_regimes)
    px_7 = _paper_target_map(args.track, 24 * 7, targets_filter)
    px_30 = _paper_target_map(args.track, 24 * 30, targets_filter)
    matched_7 = sorted(set(bt_7.keys()) & set(px_7.keys()))
    matched_30 = sorted(set(bt_30.keys()) & set(px_30.keys()))

    if len(matched_30) < 2:
        out = {
            "status": "insufficient_observation",
            "passed": False,
            "evaluated_at": started_at.isoformat(),
            "window_start": (started_at - timedelta(days=30)).isoformat(),
            "window_end": started_at.isoformat(),
            "completed_runs": completed_runs,
            "score_source": score_source,
            "min_completed_runs": args.min_completed_runs,
            "reason": "insufficient_matched_targets",
            "matched_targets_count": len(matched_30),
            "paper_filled_orders_count": 0,
            "comparison_basis": "matched_filled_orders",
        }
        print(json.dumps(out, ensure_ascii=False))
        record_metrics_test(
            test_name="check_backtest_paper_parity",
            payload=out,
            window_start=str(out.get("window_start")),
            window_end=str(out.get("window_end")),
            extra={"argv": {"track": args.track}},
        )
        return 0

    bt_avg_7, px_avg_7, rel_7, orders_7 = _window_pair(bt_7, px_7, matched_7, args.parity_floor)
    bt_avg_30, px_avg_30, rel_30, orders_30 = _window_pair(bt_30, px_30, matched_30, args.parity_floor)

    if orders_30 < 50:
        out = {
            "status": "insufficient_observation",
            "passed": False,
            "evaluated_at": started_at.isoformat(),
            "window_start": (started_at - timedelta(days=30)).isoformat(),
            "window_end": started_at.isoformat(),
            "completed_runs": completed_runs,
            "score_source": score_source,
            "min_completed_runs": args.min_completed_runs,
            "reason": "insufficient_paper_orders",
            "matched_targets_count": len(matched_30),
            "paper_filled_orders_count": orders_30,
            "comparison_basis": "matched_filled_orders",
        }
        print(json.dumps(out, ensure_ascii=False))
        record_metrics_test(
            test_name="check_backtest_paper_parity",
            payload=out,
            window_start=str(out.get("window_start")),
            window_end=str(out.get("window_end")),
            extra={"argv": {"track": args.track}},
        )
        return 0

    passed = rel_30 <= args.max_deviation
    out = {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "evaluated_at": started_at.isoformat(),
        "window_start": (started_at - timedelta(days=30)).isoformat(),
        "window_end": started_at.isoformat(),
        "track": args.track,
        "score_source": score_source,
        "max_deviation": args.max_deviation,
        "include_sources": include_sources,
        "exclude_sources": exclude_sources,
        "data_regimes": data_regimes,
        "targets_filter": targets_filter,
        "completed_runs": completed_runs,
        "matched_targets_count": len(matched_30),
        "paper_filled_orders_count": orders_30,
        "comparison_basis": "matched_filled_orders",
        "window_details": {
            "7d": {
                "backtest_avg_pnl_after_cost": bt_avg_7,
                "paper_realized_return": px_avg_7,
                "matched_targets": len(matched_7),
                "paper_filled_orders": orders_7,
                "abs_delta": abs(bt_avg_7 - px_avg_7),
                "relative_deviation": rel_7,
            },
            "30d": {
                "backtest_avg_pnl_after_cost": bt_avg_30,
                "paper_realized_return": px_avg_30,
                "matched_targets": len(matched_30),
                "paper_filled_orders": orders_30,
                "abs_delta": abs(bt_avg_30 - px_avg_30),
                "relative_deviation": rel_30,
            },
        },
        "gate_window": "30d",
        "alert_window": "7d",
    }
    print(json.dumps(out, ensure_ascii=False))
    record_metrics_test(
        test_name="check_backtest_paper_parity",
        payload=out,
        window_start=str(out.get("window_start")),
        window_end=str(out.get("window_end")),
        extra={
            "argv": {
                    "track": args.track,
                    "score_source": score_source,
                    "max_deviation": args.max_deviation,
                "min_completed_runs": args.min_completed_runs,
                    "parity_floor": args.parity_floor,
                    "include_sources": include_sources,
                    "exclude_sources": exclude_sources,
                    "data_regimes": data_regimes,
                    "targets": targets_filter,
                }
            },
        )
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"passed": False, "status": "failed", "error": str(exc)}), file=sys.stderr)
        raise
