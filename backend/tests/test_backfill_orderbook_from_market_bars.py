from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "backfill_orderbook_from_market_bars.py"
    spec = importlib.util.spec_from_file_location("backfill_orderbook_from_market_bars", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_symbols_deduplicates_and_normalizes():
    mod = _load_module()
    out = mod._parse_symbols("btc, ETH, $btc,sol,,ETH")
    assert out == ["BTC", "ETH", "SOL"]


def test_parse_dt_utc_supports_z_suffix():
    mod = _load_module()
    dt = mod._parse_dt_utc("2026-02-19T00:00:00Z")
    assert dt.tzinfo is not None
    assert dt.isoformat().startswith("2026-02-19T00:00:00")
