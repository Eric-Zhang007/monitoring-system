#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _git_meta() -> Dict[str, Any]:
    def _run(cmd: List[str]) -> str:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=2.0)
            return str(out).strip()
        except Exception:
            return ""

    head = _run(["git", "rev-parse", "HEAD"])
    short = _run(["git", "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain"]))
    return {
        "head": head,
        "head_short": short,
        "branch": branch,
        "dirty": dirty,
    }


def _summarize(metrics_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pnl: List[float] = []
    sharpe: List[float] = []
    mdd: List[float] = []
    obs_days: List[float] = []
    hit_rate: List[float] = []
    completed = 0
    failed = 0
    pnl_samples: List[float] = []
    by_horizon: Dict[str, Dict[str, float]] = {}
    by_bucket: Dict[str, Dict[str, float]] = {}
    by_symbol: Dict[str, Dict[str, float]] = {}

    for row in metrics_rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        config = row.get("config") if isinstance(row.get("config"), dict) else {}
        status = str(metrics.get("status") or "").strip().lower()
        if status == "completed":
            completed += 1
        else:
            failed += 1
            continue
        cur_pnl = _safe_float(metrics.get("pnl_after_cost"))
        pnl.append(cur_pnl)
        pnl_samples.append(cur_pnl)
        sharpe.append(_safe_float(metrics.get("sharpe_daily", metrics.get("sharpe"))))
        mdd.append(_safe_float(metrics.get("max_drawdown")))
        obs_days.append(_safe_float(metrics.get("observation_days")))
        hit_rate.append(_safe_float(metrics.get("hit_rate")))
        cur_turnover = _safe_float(metrics.get("turnover"))
        cur_ic = _safe_float(metrics.get("ic"))
        horizon = str(config.get("horizon") or "unknown").strip().lower() or "unknown"
        b_h = by_horizon.setdefault(horizon, {"pnl": 0.0, "turnover": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0, "ic": 0.0, "count": 0.0})
        b_h["pnl"] += cur_pnl
        b_h["turnover"] += cur_turnover
        b_h["max_drawdown"] = max(b_h["max_drawdown"], _safe_float(metrics.get("max_drawdown")))
        b_h["hit_rate"] += _safe_float(metrics.get("hit_rate"))
        b_h["ic"] += cur_ic
        b_h["count"] += 1.0
        for bucket, block in (metrics.get("bucket_stats") or {}).items() if isinstance(metrics.get("bucket_stats"), dict) else []:
            b = by_bucket.setdefault(str(bucket), {"pnl": 0.0, "turnover": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0, "ic": 0.0, "count": 0.0})
            b["pnl"] += _safe_float((block or {}).get("pnl_after_cost"))
            b["turnover"] += _safe_float((block or {}).get("turnover"))
            b["max_drawdown"] = max(b["max_drawdown"], _safe_float((block or {}).get("max_drawdown")))
            b["hit_rate"] += _safe_float((block or {}).get("hit_rate"))
            b["ic"] += _safe_float((block or {}).get("ic"))
            b["count"] += 1.0
        for symbol, block in (metrics.get("per_target") or {}).items() if isinstance(metrics.get("per_target"), dict) else []:
            s = by_symbol.setdefault(str(symbol).upper(), {"pnl": 0.0, "count": 0.0})
            if isinstance(block, dict):
                s["pnl"] += _safe_float(block.get("pnl_after_cost"))
            else:
                s["pnl"] += _safe_float(block)
            s["count"] += 1.0

    n = len(pnl)
    if n <= 0:
        return {
            "runs_total": len(metrics_rows),
            "runs_completed": 0,
            "runs_failed": failed,
            "mean_pnl_after_cost": 0.0,
            "mean_sharpe_daily": 0.0,
            "max_drawdown_worst": 1.0,
            "mean_observation_days": 0.0,
            "mean_hit_rate": 0.0,
            "tail_pnl_worst_1pct": 0.0,
            "horizon_stats": {},
            "bucket_stats": {},
            "symbol_stats": {},
        }
    sorted_pnl = sorted(pnl_samples)
    tail_n = max(1, int(round(len(sorted_pnl) * 0.01)))
    tail_worst = float(sum(sorted_pnl[:tail_n]) / max(1, tail_n))
    horizon_stats = {
        h: {
            "count": int(v["count"]),
            "net_pnl_after_cost": round(v["pnl"], 10),
            "turnover": round(v["turnover"] / max(1.0, v["count"]), 10),
            "max_drawdown": round(v["max_drawdown"], 10),
            "hit_rate": round(v["hit_rate"] / max(1.0, v["count"]), 10),
            "ic": round(v["ic"] / max(1.0, v["count"]), 10),
        }
        for h, v in sorted(by_horizon.items())
    }
    bucket_stats = {
        h: {
            "count": int(v["count"]),
            "net_pnl_after_cost": round(v["pnl"], 10),
            "turnover": round(v["turnover"] / max(1.0, v["count"]), 10),
            "max_drawdown": round(v["max_drawdown"], 10),
            "hit_rate": round(v["hit_rate"] / max(1.0, v["count"]), 10),
            "ic": round(v["ic"] / max(1.0, v["count"]), 10),
        }
        for h, v in sorted(by_bucket.items())
    }
    symbol_stats = {
        s: {
            "count": int(v["count"]),
            "net_pnl_after_cost": round(v["pnl"], 10),
        }
        for s, v in sorted(by_symbol.items())
    }

    return {
        "runs_total": len(metrics_rows),
        "runs_completed": completed,
        "runs_failed": failed,
        "mean_pnl_after_cost": round(sum(pnl) / n, 10),
        "mean_sharpe_daily": round(sum(sharpe) / n, 10),
        "max_drawdown_worst": round(max(mdd), 10),
        "mean_observation_days": round(sum(obs_days) / n, 6),
        "mean_hit_rate": round(sum(hit_rate) / n, 10),
        "tail_pnl_worst_1pct": round(tail_worst, 10),
        "horizon_stats": horizon_stats,
        "bucket_stats": bucket_stats,
        "symbol_stats": symbol_stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate training profitability from recent backtest runs")
    ap.add_argument("--database-url", default=DATABASE_URL)
    ap.add_argument("--track", default="liquid")
    ap.add_argument("--lookback-hours", type=int, default=int(os.getenv("PROFIT_GATE_LOOKBACK_HOURS", "168")))
    ap.add_argument("--limit", type=int, default=int(os.getenv("PROFIT_GATE_LIMIT", "32")))
    ap.add_argument("--score-source", default=os.getenv("PROFIT_GATE_SCORE_SOURCE", "model"))
    ap.add_argument("--data-regime", default=os.getenv("PROFIT_GATE_DATA_REGIME", "prod_live"))
    ap.add_argument("--min-completed-runs", type=int, default=int(os.getenv("PROFIT_GATE_MIN_COMPLETED_RUNS", "4")))
    ap.add_argument("--min-mean-pnl-after-cost", type=float, default=float(os.getenv("PROFIT_GATE_MIN_MEAN_PNL_AFTER_COST", "0.0")))
    ap.add_argument("--min-mean-sharpe-daily", type=float, default=float(os.getenv("PROFIT_GATE_MIN_MEAN_SHARPE_DAILY", "0.2")))
    ap.add_argument("--max-worst-drawdown", type=float, default=float(os.getenv("PROFIT_GATE_MAX_WORST_DRAWDOWN", "0.35")))
    ap.add_argument("--min-mean-hit-rate", type=float, default=float(os.getenv("PROFIT_GATE_MIN_MEAN_HIT_RATE", "0.48")))
    ap.add_argument(
        "--baseline-mean-pnl-after-cost",
        type=float,
        default=float(os.getenv("PROFIT_GATE_BASELINE_MEAN_PNL_AFTER_COST", "0.0")),
    )
    ap.add_argument(
        "--min-improved-wf-windows",
        type=int,
        default=int(os.getenv("PROFIT_GATE_MIN_IMPROVED_WF_WINDOWS", "2")),
    )
    ap.add_argument("--out-json", default=os.getenv("PROFIT_GATE_SNAPSHOT", "artifacts/ops/training_profitability_gate.json"))
    args = ap.parse_args()

    track = str(args.track).strip().lower()
    score_source = str(args.score_source).strip().lower()
    data_regime = str(args.data_regime).strip()

    rows: List[Dict[str, Any]] = []
    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, metrics, config
                FROM backtest_runs
                WHERE track = %s
                  AND created_at > NOW() - make_interval(hours => %s)
                  AND COALESCE(config->>'score_source', 'heuristic') = %s
                  AND COALESCE(config->>'data_regime', 'prod_live') = %s
                  AND superseded_by_run_id IS NULL
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (track, int(args.lookback_hours), score_source, data_regime, int(args.limit)),
            )
            rows = [dict(r) for r in (cur.fetchall() or [])]

    summary = _summarize(rows)
    improved_wf_windows = 0
    baseline_pnl = float(args.baseline_mean_pnl_after_cost)
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        wf = metrics.get("walk_forward_windows")
        if isinstance(wf, list):
            for w in wf:
                if not isinstance(w, dict):
                    continue
                if _safe_float(w.get("pnl_after_cost")) > baseline_pnl:
                    improved_wf_windows += 1
    if improved_wf_windows == 0 and _safe_float(summary.get("mean_pnl_after_cost")) > baseline_pnl:
        improved_wf_windows = 1

    checks = {
        "completed_runs": bool(_safe_int(summary.get("runs_completed")) >= int(args.min_completed_runs)),
        "mean_pnl_after_cost": bool(_safe_float(summary.get("mean_pnl_after_cost")) >= float(args.min_mean_pnl_after_cost)),
        "mean_sharpe_daily": bool(_safe_float(summary.get("mean_sharpe_daily")) >= float(args.min_mean_sharpe_daily)),
        "worst_drawdown": bool(_safe_float(summary.get("max_drawdown_worst"), 1.0) <= float(args.max_worst_drawdown)),
        "mean_hit_rate": bool(_safe_float(summary.get("mean_hit_rate")) >= float(args.min_mean_hit_rate)),
        "wf_windows_vs_baseline": bool(int(improved_wf_windows) >= int(max(0, args.min_improved_wf_windows))),
    }
    passed = bool(all(checks.values()))

    out = {
        "status": "passed" if passed else "failed",
        "passed": bool(passed),
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "track": track,
        "lookback_hours": int(args.lookback_hours),
        "score_source": score_source,
        "data_regime": data_regime,
        "limit": int(args.limit),
        "summary": summary,
        "improved_wf_windows": int(improved_wf_windows),
        "checks": checks,
        "thresholds": {
            "min_completed_runs": int(args.min_completed_runs),
            "min_mean_pnl_after_cost": float(args.min_mean_pnl_after_cost),
            "min_mean_sharpe_daily": float(args.min_mean_sharpe_daily),
            "max_worst_drawdown": float(args.max_worst_drawdown),
            "min_mean_hit_rate": float(args.min_mean_hit_rate),
            "baseline_mean_pnl_after_cost": float(baseline_pnl),
            "min_improved_wf_windows": int(max(0, args.min_improved_wf_windows)),
        },
        "config": {
            "database_url_source": "arg_or_env",
            "window": {
                "lookback_hours": int(args.lookback_hours),
                "limit": int(args.limit),
            },
        },
        "git": _git_meta(),
        "recent_run_ids": [int(r.get("id") or 0) for r in rows if int(r.get("id") or 0) > 0][:20],
    }

    out_path = Path(str(args.out_json)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
