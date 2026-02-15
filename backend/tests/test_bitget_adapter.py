from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import execution_engine as eng  # noqa: E402
from execution_engine import BitgetLiveAdapter, ExecutionEngine  # noqa: E402


class _Resp:
    def __init__(self, code: int, body: dict):
        self.status_code = code
        self._body = body
        self.content = b"{}"

    def json(self):
        return self._body


def test_bitget_live_missing_credentials(monkeypatch):
    monkeypatch.delenv("BITGET_API_KEY", raising=False)
    monkeypatch.delenv("BITGET_API_SECRET", raising=False)
    monkeypatch.delenv("BITGET_API_PASSPHRASE", raising=False)
    adapter = BitgetLiveAdapter()
    out = adapter.execute({"target": "BTC", "quantity": 1.0, "side": "buy"}, context={"venue": "bitget"})
    assert out["status"] == "rejected"
    assert out["reject_reason"] == "bitget_credentials_not_configured"


def test_bitget_live_submit_and_poll_success(monkeypatch):
    monkeypatch.setenv("BITGET_API_KEY", "k")
    monkeypatch.setenv("BITGET_API_SECRET", "s")
    monkeypatch.setenv("BITGET_API_PASSPHRASE", "p")

    calls = {"n": 0}

    def _fake_request(method, url, headers=None, data=None, timeout=None):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(200, {"code": "00000", "data": {"orderId": "oid-1"}})
        return _Resp(200, {"code": "00000", "data": {"status": "filled", "filledQty": "1", "avgPrice": "100"}})

    monkeypatch.setattr(eng.requests, "request", _fake_request)
    adapter = BitgetLiveAdapter()
    out = adapter.execute(
        {"target": "BTC", "quantity": 1.0, "side": "buy", "est_price": 100.0},
        context={"venue": "bitget", "market_type": "spot", "time_in_force": "IOC"},
    )
    assert out["status"] == "filled"
    assert out["venue_order_id"] == "oid-1"
    assert isinstance(out.get("lifecycle"), list)
    assert out["lifecycle"][0]["event"] == "submit"


def test_execution_engine_registers_bitget_adapter():
    engine = ExecutionEngine()
    assert "bitget_live" in engine.adapters
