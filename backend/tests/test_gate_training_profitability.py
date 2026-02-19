from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "gate_training_profitability.py"
    spec = importlib.util.spec_from_file_location("gate_training_profitability", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_handles_empty_rows():
    mod = _load_module()
    out = mod._summarize([])
    assert out["runs_total"] == 0
    assert out["runs_completed"] == 0
    assert out["max_drawdown_worst"] == 1.0


def test_summarize_uses_completed_only():
    mod = _load_module()
    rows = [
        {"metrics": {"status": "completed", "pnl_after_cost": 0.01, "sharpe_daily": 0.5, "max_drawdown": 0.2, "observation_days": 10, "hit_rate": 0.6}},
        {"metrics": {"status": "failed", "pnl_after_cost": -1}},
    ]
    out = mod._summarize(rows)
    assert out["runs_total"] == 2
    assert out["runs_completed"] == 1
    assert out["runs_failed"] == 1
    assert out["mean_pnl_after_cost"] == 0.01
