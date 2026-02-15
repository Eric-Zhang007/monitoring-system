#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from typing import Any, Dict

import requests


API_BASE = "http://localhost:8000"


def _run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def _api_status(path: str) -> int:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=8)
        return int(r.status_code)
    except Exception:
        return 0


def _recover_checks() -> Dict[str, Any]:
    time.sleep(1)
    checks = {
        "health": _api_status("/health"),
        "metrics": _api_status("/metrics"),
        "risk_limits": _api_status("/api/v2/risk/limits"),
        "pnl_attribution": _api_status("/api/v2/metrics/pnl-attribution?track=liquid&lookback_hours=24"),
        "dq_stats": _api_status("/api/v2/data-quality/stats?lookback_days=7"),
    }
    ok_count = sum(1 for v in checks.values() if v == 200)
    ratio = ok_count / max(1, len(checks))
    return {
        "checks": checks,
        "key_api_success_ratio": ratio,
        "threshold": 0.8,
        "passed": ratio >= 0.8,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Chaos drill helper")
    ap.add_argument("scenario", choices=["redis_interrupt", "db_slow", "exchange_jitter", "model_degrade", "recover"])
    args = ap.parse_args()

    actions = []
    validation: Dict[str, Any] = {}
    if args.scenario == "redis_interrupt":
        actions.append(_run(["docker", "compose", "stop", "redis"]))
    elif args.scenario == "db_slow":
        actions.append(_run(["docker", "compose", "exec", "-T", "postgres", "psql", "-U", "monitor", "-d", "monitor", "-c", "SELECT pg_sleep(5);"]))
    elif args.scenario == "exchange_jitter":
        actions.append(_run(["docker", "compose", "exec", "-T", "backend", "sh", "-lc", "export COINBASE_EXEC_TIMEOUT_SEC=0.2; echo jitter_injected"]))
    elif args.scenario == "model_degrade":
        actions.append(_run(["docker", "compose", "exec", "-T", "backend", "sh", "-lc", "mv /app/models /app/models.bak 2>/dev/null || true; echo model_artifact_hidden"]))
    else:
        actions.append(_run(["docker", "compose", "up", "-d", "redis"]))
        actions.append(_run(["docker", "compose", "exec", "-T", "backend", "sh", "-lc", "[ -d /app/models.bak ] && mv /app/models.bak /app/models || true; echo recovered"]))
        validation = _recover_checks()

    out = {
        "scenario": args.scenario,
        "steps": [{"code": c, "stdout": o, "stderr": e} for (c, o, e) in actions],
        "ok": all(c == 0 for c, _, _ in actions),
    }
    if validation:
        out["validation"] = validation
        out["ok"] = out["ok"] and bool(validation.get("passed", False))
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
