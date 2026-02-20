from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def score_source_filter(value: Optional[str]) -> str:
    v = str(value or "").strip().lower()
    return "heuristic" if v == "heuristic" else "model"


def evaluate_gate(
    *,
    track: str,
    min_ic: float,
    min_pnl_after_cost: float,
    max_drawdown: float,
    windows: int,
    score_source: str,
    repo: Any,
    env_flag_fn: Callable[[str, str], bool],
    run_source_filters_fn: Callable[[], Tuple[List[str], List[str]]],
    data_regime_filters_fn: Callable[[], List[str]],
) -> Tuple[bool, str, Dict[str, float], int]:
    score_source = score_source_filter(score_source)
    require_leakage_pass = env_flag_fn("GATE_REQUIRE_LEAKAGE_PASS", "1")
    allow_degraded_runs = env_flag_fn("GATE_ALLOW_DEGRADED_RUNS", "0")
    include_sources, exclude_sources = run_source_filters_fn()
    data_regimes = data_regime_filters_fn()
    runs = repo.list_recent_backtest_runs(
        track=track,
        limit=windows,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        data_regimes=data_regimes,
    )
    usable: List[Dict[str, Any]] = []
    for r in runs:
        metrics = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
        if not metrics:
            continue
        if r.get("superseded_by_run_id") is not None:
            continue
        if str(metrics.get("status") or "").lower() != "completed":
            continue
        if score_source_filter((r.get("config") or {}).get("score_source")) != score_source:
            continue
        if score_source == "model" and float(metrics.get("model_inference_coverage", 0.0) or 0.0) < 1.0:
            continue
        leakage_passed = bool((metrics.get("leakage_checks") or {}).get("passed", False))
        degraded = bool(metrics.get("degraded", False))
        if require_leakage_pass and not leakage_passed:
            continue
        if (not allow_degraded_runs) and degraded:
            continue
        usable.append(r)
    if len(usable) < windows:
        return False, "insufficient_windows", {"ic": 0.0, "pnl_after_cost": 0.0, "max_drawdown": 1.0}, len(usable)

    ic_vals = [float(r["metrics"].get("ic", 0.0)) for r in usable[:windows]]
    pnl_vals = [float(r["metrics"].get("pnl_after_cost", 0.0)) for r in usable[:windows]]
    dd_vals = [float(r["metrics"].get("max_drawdown", 1.0)) for r in usable[:windows]]

    summary = {
        "ic": round(float(np.mean(ic_vals)), 6),
        "pnl_after_cost": round(float(np.mean(pnl_vals)), 6),
        "max_drawdown": round(float(np.mean(dd_vals)), 6),
    }
    passed = summary["ic"] >= min_ic and summary["pnl_after_cost"] >= min_pnl_after_cost and summary["max_drawdown"] <= max_drawdown
    reason = "passed" if passed else "threshold_failed"
    return passed, reason, summary, windows
