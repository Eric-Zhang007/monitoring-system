#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any, Dict


def _run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def _json_cmd(cmd: list[str]) -> Dict[str, Any]:
    code, out, _ = _run(cmd)
    if code != 0 or not out:
        return {}
    try:
        return json.loads(out)
    except Exception:
        return {}


def main() -> int:
    min_sharpe_daily = float(os.getenv("GPU_CUTOVER_MIN_SHARPE_DAILY", os.getenv("BACKTEST_GATE_MIN_SHARPE_DAILY", "1.5")))
    strict_common = [
        "--score-source",
        "model",
        "--include-sources",
        "prod",
        "--exclude-sources",
        "smoke,async_test,maintenance",
        "--data-regimes",
        "prod_live",
    ]
    hard = _json_cmd(
        [
            "python3",
            "scripts/evaluate_hard_metrics.py",
            "--track",
            "liquid",
            "--lookback-days",
            "180",
            "--min-completed-runs",
            "5",
            "--min-observation-days",
            "14",
            "--min-sharpe-daily",
            str(min_sharpe_daily),
            *strict_common,
        ]
    ) or {}
    parity_raw = _run(
        [
            "python3",
            "scripts/check_backtest_paper_parity.py",
            "--track",
            "liquid",
            "--max-deviation",
            "0.10",
            "--min-completed-runs",
            "5",
            *strict_common,
        ]
    )[1]
    try:
        parity = json.loads(parity_raw) if parity_raw else {}
    except Exception:
        parity = {}
    contracts = _json_cmd(
        [
            "python3",
            "scripts/validate_backtest_contracts.py",
            "--track",
            "liquid",
            "--lookback-days",
            "180",
            "--score-source",
            "model",
            "--include-sources",
            "prod",
            "--exclude-sources",
            "smoke,async_test,maintenance",
            "--data-regimes",
            "prod_live",
            "--min-valid",
            "20",
        ]
    ) or {}
    no_leakage = _json_cmd(
        [
            "python3",
            "scripts/validate_no_leakage.py",
            "--track",
            "liquid",
            "--lookback-days",
            "180",
            "--score-source",
            "model",
            "--include-sources",
            "prod",
            "--exclude-sources",
            "smoke,async_test,maintenance",
            "--data-regimes",
            "prod_live",
        ]
    ) or {}

    reject_rate_raw = hard.get("execution_reject_rate")
    reject_rate = float(reject_rate_raw) if reject_rate_raw is not None else 1.0
    completed = int(hard.get("samples_completed") or 0)
    artifact_ratio_raw = hard.get("artifact_failure_ratio")
    artifact_ratio = float(artifact_ratio_raw) if artifact_ratio_raw is not None else 1.0
    hard_passed = bool(hard.get("hard_passed"))
    parity_passed = bool(parity.get("passed")) and str(parity.get("status") or "") == "passed"
    contracts_passed = bool(contracts.get("passed"))
    leakage_passed = bool(no_leakage.get("passed"))

    gates = {
        "strict_contract_passed": contracts_passed,
        "hard_metrics_passed": hard_passed,
        "parity_30d_passed": parity_passed,
        "no_leakage_passed": leakage_passed,
        "artifact_failure_ratio_le_0_05": artifact_ratio <= 0.05,
        "samples_completed_ge_20": completed >= 20,
        "execution_reject_rate_lt_0_01": reject_rate < 0.01,
    }
    ready = all(gates.values())
    blockers = [k for k, v in gates.items() if not bool(v)]

    now = datetime.now(timezone.utc)
    suggested_start = (now + timedelta(days=5)).date().isoformat()
    suggested_end = (now + timedelta(days=7)).date().isoformat()

    out = {
        "ready_for_gpu_cutover": ready,
        "gates": gates,
        "blockers": blockers,
        "strict_filters": {
            "run_source": "prod",
            "score_source": "model",
            "data_regime": "prod_live",
        },
        "hard_gate_config": {
            "min_sharpe_daily": float(min_sharpe_daily),
            "min_completed_runs": 5,
            "min_observation_days": 14,
        },
        "contracts": contracts,
        "hard_metrics": hard,
        "parity": parity,
        "no_leakage": no_leakage,
        "recommended_window": f"{suggested_start} to {suggested_end}" if ready else "defer_until_gates_green",
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
