from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_prod_live_backtest_batch.py"
    spec = importlib.util.spec_from_file_location("run_prod_live_backtest_batch", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_cost_floor_blocks_too_optimistic_defaults(monkeypatch):
    mod = _load_module()
    monkeypatch.setenv("BACKTEST_MIN_FEE_BPS", "3.0")
    monkeypatch.setenv("BACKTEST_MIN_SLIPPAGE_BPS", "2.0")
    with pytest.raises(RuntimeError):
        mod._validate_cost_floor(0.5, 0.2)


def test_validate_cost_floor_allows_realistic_costs(monkeypatch):
    mod = _load_module()
    monkeypatch.setenv("BACKTEST_MIN_FEE_BPS", "3.0")
    monkeypatch.setenv("BACKTEST_MIN_SLIPPAGE_BPS", "2.0")
    mod._validate_cost_floor(5.0, 3.0)
