from __future__ import annotations

from pathlib import Path
import sys
import pytest

_collector_path = Path(__file__).resolve().parents[2] / "collector"
if _collector_path.exists():
    sys.path.append(str(_collector_path))
else:
    pytest.skip("collector module not available in this test runtime", allow_module_level=True)

from entity_linking import extract_entities  # noqa: E402


def test_extract_entities_alias_mapping_for_liquid_symbols():
    ents = extract_entities("Bitcoin and XBT lead, Ethereum and Solana follow")
    symbols = {str(e.get("symbol") or "").upper() for e in ents}
    assert "BTC" in symbols
    assert "ETH" in symbols
    assert "SOL" in symbols
    assert "XBT" not in symbols
