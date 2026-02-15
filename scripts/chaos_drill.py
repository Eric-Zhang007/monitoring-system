#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess


def _run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Chaos drill helper")
    ap.add_argument("scenario", choices=["redis_interrupt", "db_slow", "exchange_jitter", "model_degrade", "recover"])
    args = ap.parse_args()

    actions = []
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

    out = {
        "scenario": args.scenario,
        "steps": [
            {"code": c, "stdout": o, "stderr": e}
            for (c, o, e) in actions
        ],
        "ok": all(c == 0 for c, _, _ in actions),
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
