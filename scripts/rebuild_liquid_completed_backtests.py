#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any, Dict, List

import requests


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay failed liquid backtests to rebuild completed samples")
    ap.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    sql = (
        "SELECT id, COALESCE(config::text,'{}') FROM backtest_runs "
        "WHERE track='liquid' "
        "AND COALESCE(metrics->>'status','')='failed' "
        "AND COALESCE(metrics->>'reason','')='model_artifact_missing' "
        "AND superseded_by_run_id IS NULL "
        "ORDER BY id DESC "
        f"LIMIT {max(1, args.limit)};"
    )
    rows = [r for r in _run_psql(sql).splitlines() if r.strip()]

    attempted = 0
    submitted = 0
    superseded = 0
    errors: List[Dict[str, Any]] = []
    for row in rows:
        attempted += 1
        parts = row.split("|", 1)
        if len(parts) != 2:
            continue
        run_id = int(parts[0])
        try:
            cfg = json.loads(parts[1])
        except Exception:
            cfg = {}
        payload = {
            "track": cfg.get("track") or "liquid",
            "run_source": "maintenance",
            "targets": cfg.get("targets") or ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "TRX", "AVAX", "LINK"],
            "horizon": cfg.get("horizon") or "1d",
            "model_name": cfg.get("model_name") or "liquid_ttm_ensemble",
            "model_version": cfg.get("model_version") or "v2.1",
            "data_version": cfg.get("data_version") or "v1",
            "lookback_days": int(cfg.get("lookback_days") or 90),
            "train_days": int(cfg.get("train_days") or 35),
            "test_days": int(cfg.get("test_days") or 7),
            "fee_bps": float(cfg.get("fee_bps") or 5.0),
            "slippage_bps": float(cfg.get("slippage_bps") or 3.0),
        }
        try:
            resp = requests.post(f"{args.api_base}/api/v2/backtest/run", json=payload, timeout=60)
            if resp.status_code < 300:
                body = resp.json()
                submitted += 1
                if str(body.get("status") or "").lower() == "completed":
                    new_run_id = int(body.get("run_id") or 0)
                    if new_run_id > 0:
                        upd_sql = (
                            "UPDATE backtest_runs "
                            f"SET superseded_by_run_id={new_run_id}, supersede_reason='artifact_replayed_completed', superseded_at=NOW() "
                            f"WHERE id={run_id};"
                        )
                        _run_psql(upd_sql)
                        superseded += 1
            else:
                errors.append({"source_run_id": run_id, "status_code": resp.status_code, "body": resp.text[:200]})
        except Exception as exc:
            errors.append({"source_run_id": run_id, "error": type(exc).__name__})

    out = {
        "attempted": attempted,
        "submitted": submitted,
        "superseded": superseded,
        "failed": max(0, attempted - submitted),
        "errors": errors[:20],
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0 if submitted > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
