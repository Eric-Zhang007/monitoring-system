from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "ingest_binance_derivatives_signals.py"
    spec = importlib.util.spec_from_file_location("ingest_binance_derivatives_signals", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_symbols_deduplicates_and_normalizes():
    mod = _load_module()
    out = mod._parse_symbols("btc, ETH, $btc,sol,,ETH")
    assert out == ["BTC", "ETH", "SOL"]


def test_parse_symbol_map_defaults_to_usdt_pair():
    mod = _load_module()
    out = mod._parse_symbol_map("BTC,ETH:ETHUSD_PERP")
    assert out["BTC"] == "BTCUSDT"
    assert out["ETH"] == "ETHUSD_PERP"


def test_dedup_points_keeps_latest_value_for_same_ts():
    mod = _load_module()
    points = [(1000, 1.0), (1000, 2.0), (2000, 3.0)]
    out = mod._dedup_points(points)
    assert out == [(1000, 2.0), (2000, 3.0)]
