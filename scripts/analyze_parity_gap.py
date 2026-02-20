#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from _psql import run_psql


def _run_psql(sql: str) -> str:
    return run_psql(sql)


def _parse_sources(raw: str) -> List[str]:
    vals: List[str] = []
    for p in (raw or "").split(","):
        s = p.strip().lower()
        if s and re.fullmatch(r"[a-z0-9_-]+", s):
            vals.append(s)
    return vals


def _parse_regimes(raw: str) -> List[str]:
    allowed = {"prod_live", "maintenance_replay", "mixed"}
    return [p.strip().lower() for p in (raw or "").split(",") if p.strip().lower() in allowed]


def _sql_source_filters(include_sources: List[str], exclude_sources: List[str]) -> str:
    parts: List[str] = []
    if include_sources:
        parts.append("COALESCE(run_source,'prod') IN (" + ",".join(f"'{s}'" for s in include_sources) + ")")
    if exclude_sources:
        parts.append("COALESCE(run_source,'prod') NOT IN (" + ",".join(f"'{s}'" for s in exclude_sources) + ")")
    return (" AND " + " AND ".join(parts)) if parts else ""


def _sql_regime_filter(regimes: List[str]) -> str:
    if not regimes:
        return ""
    return " AND COALESCE(NULLIF(config->>'data_regime',''),'missing') IN (" + ",".join(f"'{r}'" for r in regimes) + ")"


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _bt_target_map(
    track: str,
    hours: int,
    score_source: str,
    include_sources: List[str],
    exclude_sources: List[str],
    regimes: List[str],
) -> Dict[str, Dict[str, float]]:
    sql = (
        "SELECT config::text, metrics::text FROM backtest_runs "
        f"WHERE track='{track}' "
        "AND COALESCE(metrics->>'status','')='completed' "
        f"AND COALESCE(config->>'score_source','heuristic')='{score_source}' "
        "AND superseded_by_run_id IS NULL "
        f"AND created_at > NOW() - make_interval(hours => {int(hours)}) "
        + _sql_source_filters(include_sources, exclude_sources)
        + _sql_regime_filter(regimes)
        + ";"
    )
    rows = [r for r in _run_psql(sql).splitlines() if r.strip()]
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        parts = row.split("|", 1)
        if len(parts) != 2:
            continue
        try:
            cfg = json.loads(parts[0])
            metrics = json.loads(parts[1])
        except Exception:
            continue
        run_targets = cfg.get("targets") if isinstance(cfg.get("targets"), list) else []
        run_targets = [str(t).upper() for t in run_targets if str(t).strip()]
        per_target = metrics.get("per_target") if isinstance(metrics.get("per_target"), dict) else {}
        tkeys = [str(t).upper() for t in per_target.keys()]
        effective_targets = tkeys or run_targets
        if not effective_targets:
            continue
        n_targets = float(max(1, len(effective_targets)))
        costs = metrics.get("cost_breakdown") if isinstance(metrics.get("cost_breakdown"), dict) else {}
        fee_share = _safe_float(costs.get("fee"), default=0.0) / n_targets
        slip_share = _safe_float(costs.get("slippage"), default=0.0) / n_targets
        impact_share = _safe_float(costs.get("impact"), default=0.0) / n_targets
        global_pnl_share = _safe_float(metrics.get("pnl_after_cost"), default=0.0) / n_targets
        for t in effective_targets:
            sub = per_target.get(t)
            pnl = _safe_float((sub or {}).get("pnl_after_cost"), default=global_pnl_share) if isinstance(sub, dict) else global_pnl_share
            st = out.setdefault(
                str(t).upper(),
                {
                    "sum": 0.0,
                    "count": 0.0,
                    "cost_fee_sum": 0.0,
                    "cost_slippage_sum": 0.0,
                    "cost_impact_sum": 0.0,
                },
            )
            st["sum"] += pnl
            st["count"] += 1.0
            st["cost_fee_sum"] += fee_share
            st["cost_slippage_sum"] += slip_share
            st["cost_impact_sum"] += impact_share
    return out


def _paper_target_map(track: str, hours: int) -> Dict[str, Dict[str, float]]:
    sql = (
        "SELECT UPPER(target), side, "
        "COALESCE((metadata->'execution'->>'filled_qty')::double precision, quantity::double precision, 0.0), "
        "COALESCE((metadata->'execution'->>'avg_fill_price')::double precision, est_price::double precision, 0.0), "
        "COALESCE(est_price::double precision, 0.0), "
        "COALESCE(est_cost_bps::double precision, 0.0), "
        "COALESCE((metadata->'execution'->>'fees_paid')::double precision, 0.0) "
        "FROM orders_sim "
        f"WHERE track='{track}' "
        "AND status IN ('filled','partially_filled') "
        f"AND created_at > NOW() - make_interval(hours => {int(hours)});"
    )
    rows = [r for r in _run_psql(sql).splitlines() if r.strip()]
    out: Dict[str, Dict[str, float]] = {}
    for line in rows:
        parts = line.split("|")
        if len(parts) != 7:
            continue
        target, side, qty_raw, fill_raw, est_raw, est_cost_bps_raw, fees_paid_raw = parts
        est = _safe_float(est_raw, default=0.0)
        qty = _safe_float(qty_raw, default=0.0)
        fill = _safe_float(fill_raw, default=0.0)
        if est <= 0.0 or qty <= 0.0:
            continue
        realized = (fill - est) / max(abs(est), 1e-12)
        if side.lower() == "sell":
            realized = -realized
        adverse_slippage = max(0.0, (fill - est) / max(abs(est), 1e-12)) if side.lower() == "buy" else max(0.0, (est - fill) / max(abs(est), 1e-12))
        notional = abs(qty * est)
        fee_return = max(0.0, _safe_float(fees_paid_raw, default=0.0) / max(notional, 1e-12))
        est_cost_return = max(0.0, _safe_float(est_cost_bps_raw, default=0.0) / 10000.0)
        impact_est = max(0.0, est_cost_return - fee_return - adverse_slippage)
        st = out.setdefault(
            target,
            {
                "sum_weighted": 0.0,
                "sum_notional": 0.0,
                "orders": 0.0,
                "sum_fee": 0.0,
                "sum_slippage": 0.0,
                "sum_impact": 0.0,
            },
        )
        st["sum_weighted"] += realized * notional
        st["sum_notional"] += notional
        st["orders"] += 1.0
        st["sum_fee"] += fee_return * notional
        st["sum_slippage"] += adverse_slippage * notional
        st["sum_impact"] += impact_est * notional
    return out


def _avg_bt(d: Dict[str, float]) -> float:
    return float(d.get("sum", 0.0)) / max(1.0, float(d.get("count", 0.0)))


def _avg_paper(d: Dict[str, float]) -> float:
    return float(d.get("sum_weighted", 0.0)) / max(1e-12, float(d.get("sum_notional", 0.0)))


def _avg_cost(d: Dict[str, float], key: str) -> float:
    return float(d.get(key, 0.0)) / max(1e-12, float(d.get("sum_notional", 0.0)))


def _target_breakdown(bt_map: Dict[str, Dict[str, float]], paper_map: Dict[str, Dict[str, float]], floor: float) -> List[Dict[str, float]]:
    matched = sorted(set(bt_map.keys()) & set(paper_map.keys()))
    out: List[Dict[str, Any]] = []
    for t in matched:
        b = _avg_bt(bt_map[t])
        p = _avg_paper(paper_map[t])
        delta = abs(b - p)
        rel = delta / max(1e-6, float(floor), abs(b), abs(p))
        bt_count = max(1.0, float((bt_map[t] or {}).get("count", 0.0)))
        bt_fee = float((bt_map[t] or {}).get("cost_fee_sum", 0.0)) / bt_count
        bt_slippage = float((bt_map[t] or {}).get("cost_slippage_sum", 0.0)) / bt_count
        bt_impact = float((bt_map[t] or {}).get("cost_impact_sum", 0.0)) / bt_count
        paper_fee = _avg_cost(paper_map[t], "sum_fee")
        paper_slippage = _avg_cost(paper_map[t], "sum_slippage")
        paper_impact = _avg_cost(paper_map[t], "sum_impact")
        total_bt_cost = bt_fee + bt_slippage + bt_impact
        total_paper_cost = paper_fee + paper_slippage + paper_impact
        out.append(
            {
                "target": t,
                "backtest_avg_pnl_after_cost": round(b, 6),
                "paper_realized_return": round(p, 6),
                "abs_delta": round(delta, 6),
                "relative_deviation": round(rel, 6),
                "paper_orders": int(float((paper_map[t] or {}).get("orders", 0.0))),
                "backtest_cost_fee_est": round(bt_fee, 6),
                "backtest_cost_slippage_est": round(bt_slippage, 6),
                "backtest_cost_impact_est": round(bt_impact, 6),
                "paper_cost_fee_est": round(paper_fee, 6),
                "paper_cost_slippage_est": round(paper_slippage, 6),
                "paper_cost_impact_est": round(paper_impact, 6),
                "cost_delta_fee": round(bt_fee - paper_fee, 6),
                "cost_delta_slippage": round(bt_slippage - paper_slippage, 6),
                "cost_delta_impact": round(bt_impact - paper_impact, 6),
                "cost_delta_total": round(total_bt_cost - total_paper_cost, 6),
                "cost_estimation_method": "backtest_equal_target_split_vs_paper_execution+est_cost_bps",
            }
        )
    out.sort(key=lambda x: x["relative_deviation"], reverse=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze target-level parity gap")
    p.add_argument("--track", default="liquid")
    p.add_argument("--score-source", default="model", choices=["model", "heuristic"])
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--parity-floor", type=float, default=0.02)
    p.add_argument("--include-sources", default="prod")
    p.add_argument("--exclude-sources", default="smoke,async_test,maintenance")
    p.add_argument("--data-regimes", default="prod_live")
    args = p.parse_args()

    hours = max(1, int(args.window_days) * 24)
    include_sources = _parse_sources(args.include_sources)
    exclude_sources = _parse_sources(args.exclude_sources)
    regimes = _parse_regimes(args.data_regimes)

    bt = _bt_target_map(args.track, hours, args.score_source, include_sources, exclude_sources, regimes)
    paper = _paper_target_map(args.track, hours)
    breakdown = _target_breakdown(bt, paper, floor=float(args.parity_floor))

    out = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "window_start": (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(),
        "window_end": datetime.now(timezone.utc).isoformat(),
        "track": args.track,
        "score_source": args.score_source,
        "include_sources": include_sources,
        "exclude_sources": exclude_sources,
        "data_regimes": regimes,
        "matched_targets": len(breakdown),
        "targets": breakdown,
        "top_deviation_targets": breakdown[:5],
        "top_cost_gap_targets": sorted(breakdown, key=lambda x: abs(float(x.get("cost_delta_total", 0.0))), reverse=True)[:5],
        "notes": {
            "paper_cost_components_are_estimated": True,
            "backtest_cost_components_are_target_split_estimates": True,
        },
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
