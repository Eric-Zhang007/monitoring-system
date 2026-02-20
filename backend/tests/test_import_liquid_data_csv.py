from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "import_liquid_data_csv.py"
    spec = importlib.util.spec_from_file_location("import_liquid_data_csv", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_orderbook_rows_deduplicates_by_symbol_ts_source():
    mod = _load_module()
    rows = [
        {
            "symbol": "btc",
            "ts": "2026-02-19T00:00:00Z",
            "bid_px": "1",
            "ask_px": "2",
            "bid_sz": "3",
            "ask_sz": "4",
            "spread_bps": "5",
            "imbalance": "0.1",
            "source": "s1",
        },
        {
            "symbol": "BTC",
            "ts": "2026-02-19T00:00:00Z",
            "bid_px": "10",
            "ask_px": "20",
            "bid_sz": "30",
            "ask_sz": "40",
            "spread_bps": "50",
            "imbalance": "0.2",
            "source": "s1",
        },
    ]
    out = mod._normalize_orderbook_rows(rows)
    assert len(out) == 1
    assert out[0][0] == "BTC"


def test_parse_ts_supports_z_suffix():
    mod = _load_module()
    dt = mod._parse_ts("2026-02-19T00:00:00Z")
    assert dt.year == 2026
