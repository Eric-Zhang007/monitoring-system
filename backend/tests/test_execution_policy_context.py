from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "backend"))

from execution_policy import resolve_order_execution_context  # noqa: E402


def test_execution_policy_uses_short_horizon_for_1h():
    order = {"metadata": {"horizon": "1h"}}
    out = resolve_order_execution_context(order, {"time_in_force": "IOC"})
    assert out["execution_policy"] == "short_horizon_exec"
    assert out["horizon"] == "1h"


def test_execution_policy_uses_long_horizon_for_7d():
    order = {"metadata": {"selected_horizon": "7d"}}
    out = resolve_order_execution_context(order, {"time_in_force": "IOC"})
    assert out["execution_policy"] == "long_horizon_exec"
    assert out["horizon"] == "7d"
