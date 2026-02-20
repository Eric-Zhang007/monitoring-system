from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "export_liquid_data_csv.py"
    spec = importlib.util.spec_from_file_location("export_liquid_data_csv", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_symbols_unique_and_upper():
    mod = _load_module()
    out = mod._parse_symbols("btc,ETH,$btc,sol")
    assert out == ["BTC", "ETH", "SOL"]


def test_parse_dt_utc_supports_z_suffix():
    mod = _load_module()
    dt = mod._parse_dt_utc("2026-02-19T00:00:00Z")
    assert dt.tzinfo is not None
