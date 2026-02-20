from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "audit_required_data_readiness.py"
    spec = importlib.util.spec_from_file_location("audit_required_data_readiness", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_mode_windows_production_uses_lookback_window():
    mod = _load_module()
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 2, 20, tzinfo=timezone.utc)
    mode, lookback_start, deriv_start = mod._resolve_mode_windows(
        mode="production",
        start_dt=start,
        end_dt=end,
        lookback_days=420,
        recent_derivatives_days=30,
    )
    assert mode == "production"
    assert lookback_start == end - timedelta(days=420)
    assert deriv_start == end - timedelta(days=30)


def test_resolve_mode_windows_research_uses_full_history_start():
    mod = _load_module()
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 2, 20, tzinfo=timezone.utc)
    mode, lookback_start, deriv_start = mod._resolve_mode_windows(
        mode="research",
        start_dt=start,
        end_dt=end,
        lookback_days=420,
        recent_derivatives_days=30,
    )
    assert mode == "research"
    assert lookback_start == start
    assert deriv_start == end - timedelta(days=30)
