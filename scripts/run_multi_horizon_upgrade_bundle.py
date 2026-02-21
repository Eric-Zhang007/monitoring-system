#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_meta() -> Dict[str, Any]:
    def _run(cmd: List[str]) -> str:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=2.0)
            return str(out).strip()
        except Exception:
            return ""

    return {
        "head": _run(["git", "rev-parse", "HEAD"]),
        "head_short": _run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def _default_python_bin() -> str:
    py_env = str(os.getenv("PYTHON_BIN", "")).strip()
    if py_env:
        return py_env
    venv_py = Path(".venv/bin/python")
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _run_step(name: str, cmd: List[str], required: bool, env: Dict[str, str] | None = None) -> Dict[str, Any]:
    started = time.time()
    try:
        run_env = os.environ.copy()
        if env:
            run_env.update({str(k): str(v) for k, v in env.items()})
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=run_env)
        rc = int(proc.returncode)
        out = (proc.stdout or "")[-4000:]
        err = (proc.stderr or "")[-4000:]
    except Exception as exc:
        rc = 99
        out = ""
        err = str(exc)
    ended = time.time()
    return {
        "name": name,
        "cmd": cmd,
        "required": bool(required),
        "return_code": rc,
        "passed": bool(rc == 0),
        "duration_sec": round(max(0.0, ended - started), 6),
        "stdout_tail": out,
        "stderr_tail": err,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run reproducible multi-horizon upgrade checks and emit artifact summary")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--out-json", default="artifacts/upgrade/multi_horizon_upgrade_bundle_latest.json")
    ap.add_argument("--run-full-pytest", action="store_true")
    ap.add_argument("--skip-db-gates", action="store_true")
    ap.add_argument("--require-db-gates", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    py = _default_python_bin()
    steps: List[Dict[str, Any]] = [
        {
            "name": "targeted_pytests",
            "cmd": [
                "pytest",
                "-q",
                "backend/tests/test_phase0_cpu_smoke.py",
                "backend/tests/test_feature_contract_alignment.py",
                "backend/tests/test_multi_horizon_signal_logic.py",
                "backend/tests/test_portfolio_allocator_constraints.py",
                "backend/tests/test_execution_policy_context.py",
                "backend/tests/test_paper_trading_execution_events.py",
                "backend/tests/test_liquid_model_registry.py",
                "backend/tests/test_liquid_registry_routes.py",
                "backend/tests/test_model_router_residual_fusion.py",
            ],
            "required": True,
        }
    ]
    if args.run_full_pytest:
        steps.append({"name": "full_pytest", "cmd": ["pytest", "-q"], "required": True})

    if not args.skip_db_gates:
        steps.extend(
            [
                {
                    "name": "validate_asof_alignment",
                    "cmd": [
                        py,
                        "scripts/validate_asof_alignment.py",
                        "--database-url",
                        str(args.database_url),
                        "--out-json",
                        "artifacts/audit/asof_alignment_multi_horizon_latest.json",
                    ],
                    "required": bool(args.require_db_gates),
                },
                {
                    "name": "validate_no_leakage",
                    "cmd": [
                        py,
                        "scripts/validate_no_leakage.py",
                        "--track",
                        "liquid",
                        "--lookback-days",
                        "30",
                    ],
                    "required": bool(args.require_db_gates),
                    "env": {
                        "DATABASE_URL": str(args.database_url),
                        "METRICS_DATABASE_URL": str(args.database_url),
                    },
                },
                {
                    "name": "gate_training_profitability",
                    "cmd": [
                        py,
                        "scripts/gate_training_profitability.py",
                        "--database-url",
                        str(args.database_url),
                        "--lookback-hours",
                        "24",
                        "--limit",
                        "20",
                        "--out-json",
                        "artifacts/ops/training_profitability_gate_multi_horizon_latest.json",
                    ],
                    "required": bool(args.require_db_gates),
                },
            ]
        )

    started_at = _now_iso()
    results: List[Dict[str, Any]] = []
    for step in steps:
        results.append(
            _run_step(
                name=str(step["name"]),
                cmd=list(step["cmd"]),
                required=bool(step["required"]),
                env=step.get("env") if isinstance(step.get("env"), dict) else None,
            )
        )

    required_failures = [r for r in results if bool(r.get("required")) and not bool(r.get("passed"))]
    optional_failures = [r for r in results if (not bool(r.get("required"))) and (not bool(r.get("passed")))]
    payload = {
        "status": "passed" if not required_failures else "failed",
        "passed": bool(not required_failures),
        "started_at": started_at,
        "finished_at": _now_iso(),
        "strict": bool(args.strict),
        "database_url_source": "arg_or_env",
        "git": _git_meta(),
        "results": results,
        "summary": {
            "steps_total": len(results),
            "required_failed": len(required_failures),
            "optional_failed": len(optional_failures),
        },
    }

    out_path = Path(str(args.out_json)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if args.strict:
        return 0 if payload["passed"] else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
