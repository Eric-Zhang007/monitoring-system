from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "ingest_binance_aux_signals.py"
    spec = importlib.util.spec_from_file_location("ingest_binance_aux_signals", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_funding_rows_expands_to_timeframe_with_next_ts():
    mod = _load_module()
    raw = [
        (0, 0.001),
        (600_000, -0.002),
    ]
    rows = mod._build_funding_rows(
        raw,
        start_ms=0,
        end_ms=600_000,
        bucket_sec=300,
        expand_to_timeframe=True,
    )
    assert [r[0] for r in rows] == [0, 300_000, 600_000]
    assert rows[0][1] == 0.001
    assert rows[0][2] == 600_000
    assert rows[-1][1] == -0.002
    assert rows[-1][2] is None


def test_open_interest_proxy_uses_bucket_last_and_delta():
    mod = _load_module()
    raw = [
        (0, 10.0),
        (100_000, 12.0),
        (300_000, 11.0),
    ]
    inflow_rows, level_rows = mod._build_open_interest_proxy_rows(raw, bucket_sec=300)

    assert level_rows == [(0, 12.0), (300_000, 11.0)]
    assert inflow_rows[0] == (0, 0.0)
    assert inflow_rows[1] == (300_000, -1.0)
