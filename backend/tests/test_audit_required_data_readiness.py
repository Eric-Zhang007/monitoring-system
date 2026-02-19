from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "audit_required_data_readiness.py"
    spec = importlib.util.spec_from_file_location("audit_required_data_readiness", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timeframe_seconds_supports_minute_hour_day():
    mod = _load_module()
    assert mod._timeframe_seconds("5m") == 300
    assert mod._timeframe_seconds("1h") == 3600
    assert mod._timeframe_seconds("1d") == 86400


def test_expected_bucket_count_inclusive():
    mod = _load_module()
    start = datetime(2026, 2, 19, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 2, 19, 0, 10, tzinfo=timezone.utc)
    assert mod._expected_bucket_count(start, end, 300) == 3


def test_parse_symbols_deduplicates():
    mod = _load_module()
    out = mod._parse_symbols("btc,ETH,$btc,sol")
    assert out == ["BTC", "ETH", "SOL"]
