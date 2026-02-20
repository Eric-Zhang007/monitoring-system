from __future__ import annotations

from pathlib import Path
import sys

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from execution_engine import BitgetLiveAdapter  # noqa: E402
import v2_router as router_mod  # noqa: E402


def _base_order() -> dict:
    return {"target": "BTC", "side": "buy", "quantity": 0.01, "est_price": 50000.0}


def test_bitget_timeout_is_transport_error_category(monkeypatch):
    adapter = BitgetLiveAdapter()
    adapter.api_key = "k"
    adapter.api_secret = "s"
    adapter.api_passphrase = "p"

    def _raise_timeout(*args, **kwargs):
        raise requests.Timeout("timeout")

    monkeypatch.setattr(adapter, "_request", _raise_timeout)
    out = adapter.execute(_base_order(), context={"venue": "bitget", "market_type": "spot"})
    assert str(out.get("reject_reason")).startswith("bitget_transport_error:timeout")
    norm = router_mod._normalize_execution_payload(out)
    assert norm["reject_reason_category"] == "bitget_transport_error"


def test_bitget_connection_error_is_transport_error_category(monkeypatch):
    adapter = BitgetLiveAdapter()
    adapter.api_key = "k"
    adapter.api_secret = "s"
    adapter.api_passphrase = "p"

    def _raise_conn(*args, **kwargs):
        raise requests.ConnectionError("conn down")

    monkeypatch.setattr(adapter, "_request", _raise_conn)
    out = adapter.execute(_base_order(), context={"venue": "bitget", "market_type": "spot"})
    assert str(out.get("reject_reason")).startswith("bitget_transport_error:ConnectionError")
    norm = router_mod._normalize_execution_payload(out)
    assert norm["reject_reason_category"] == "bitget_transport_error"


def test_bitget_json_parse_error_is_unknown_error_category(monkeypatch):
    adapter = BitgetLiveAdapter()
    adapter.api_key = "k"
    adapter.api_secret = "s"
    adapter.api_passphrase = "p"

    def _raise_parse(*args, **kwargs):
        raise ValueError("bitget_response_parse_error")

    monkeypatch.setattr(adapter, "_request", _raise_parse)
    out = adapter.execute(_base_order(), context={"venue": "bitget", "market_type": "spot"})
    assert str(out.get("reject_reason")).startswith("bitget_unknown_error:ValueError")
    norm = router_mod._normalize_execution_payload(out)
    assert norm["reject_reason_category"] == "bitget_unknown_error"
