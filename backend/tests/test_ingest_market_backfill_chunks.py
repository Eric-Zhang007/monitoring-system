from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "ingest_bitget_market_bars.py"
    spec = importlib.util.spec_from_file_location("ingest_bitget_market_bars", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_iter_chunks_ms_is_non_overlapping_and_covering():
    mod = _load_module()
    # 3 days in milliseconds, chunk into 1-day windows.
    day_ms = 24 * 60 * 60 * 1000
    start_ms = 0
    end_ms = 3 * day_ms - 1
    chunks = list(mod._iter_chunks_ms(start_ms, end_ms, chunk_days=1))
    assert len(chunks) == 3
    assert chunks[0] == (0, day_ms - 1)
    assert chunks[1] == (day_ms, 2 * day_ms - 1)
    assert chunks[2] == (2 * day_ms, 3 * day_ms - 1)


def test_chunk_id_contains_index_and_window():
    mod = _load_module()
    cid = mod._chunk_id(7, 1234, 5678)
    assert cid.startswith("00007_")
    assert "1234-5678" in cid
