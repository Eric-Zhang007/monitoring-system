from __future__ import annotations

from typing import Any, Dict, List

import execution_engine as eng
from execution_engine import BitgetLiveAdapter, CoinbaseLiveAdapter


class _Resp:
    def __init__(self, status_code: int, body: Dict[str, Any]):
        self.status_code = int(status_code)
        self._body = dict(body)
        self.content = b"{}"

    def json(self):
        return dict(self._body)


def test_coinbase_request_retries_on_429(monkeypatch):
    monkeypatch.setenv("COINBASE_HTTP_MAX_RETRIES", "3")
    adapter = CoinbaseLiveAdapter()
    monkeypatch.setattr(adapter, "_build_jwt", lambda method, path: "token")

    call_count = {"n": 0}
    sleeps: List[float] = []

    def _fake_request(method, url, headers=None, json=None, params=None, timeout=None):  # noqa: ANN001
        _ = (method, url, headers, json, params, timeout)
        call_count["n"] += 1
        if call_count["n"] < 3:
            return _Resp(429, {"error": "rate_limited"})
        return _Resp(200, {"ok": True})

    monkeypatch.setattr(eng.requests, "request", _fake_request)
    monkeypatch.setattr(eng.time, "sleep", lambda sec: sleeps.append(float(sec)))

    code, body = adapter._request("GET", "/api/v3/brokerage/accounts")
    assert code == 200
    assert body.get("ok") is True
    assert call_count["n"] == 3
    assert len(sleeps) == 2


def test_bitget_request_retries_on_5xx(monkeypatch):
    monkeypatch.setenv("BITGET_HTTP_MAX_RETRIES", "3")
    adapter = BitgetLiveAdapter()

    call_count = {"n": 0}
    sleeps: List[float] = []

    def _fake_request(method, url, headers=None, data=None, timeout=None):  # noqa: ANN001
        _ = (method, url, headers, data, timeout)
        call_count["n"] += 1
        if call_count["n"] < 3:
            return _Resp(502, {"code": "error", "msg": "bad_gateway"})
        return _Resp(200, {"code": "00000", "data": {}})

    monkeypatch.setattr(eng.requests, "request", _fake_request)
    monkeypatch.setattr(eng.time, "sleep", lambda sec: sleeps.append(float(sec)))

    code, body = adapter._request("POST", "/api/v2/spot/trade/place-order", payload={"symbol": "BTCUSDT"})
    assert code == 200
    assert str(body.get("code")) == "00000"
    assert call_count["n"] == 3
    assert len(sleeps) == 2


def test_coinbase_submit_order_idempotency(monkeypatch):
    adapter = CoinbaseLiveAdapter()
    adapter.api_key_name = "k"
    adapter.api_private_key = "s"

    call_count = {"n": 0}

    def _fake_request(method: str, path: str, **kwargs):
        _ = kwargs
        call_count["n"] += 1
        if method == "POST" and path == "/api/v3/brokerage/orders":
            return 200, {"success_response": {"order_id": "cb-oid-1"}, "order": {"status": "OPEN"}}
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(adapter, "_request", _fake_request)

    child = {
        "client_order_id": "idem-coinbase-1",
        "symbol": "BTC",
        "side": "buy",
        "qty": 0.1,
        "limit_price": 100.0,
        "tif": "IOC",
    }
    venue_order_id_1, ack_1 = adapter.submit_order(child, {"time_in_force": "IOC"})
    venue_order_id_2, ack_2 = adapter.submit_order(child, {"time_in_force": "IOC"})
    assert venue_order_id_1 == "cb-oid-1"
    assert venue_order_id_2 == "cb-oid-1"
    assert call_count["n"] == 1
    assert ack_1.get("status") in {"submitted", "partially_filled", "filled", "rejected"}
    assert ack_2.get("idempotency_hit") is True


def test_bitget_submit_order_idempotency(monkeypatch):
    monkeypatch.setenv("BITGET_API_KEY", "k")
    monkeypatch.setenv("BITGET_API_SECRET", "s")
    monkeypatch.setenv("BITGET_API_PASSPHRASE", "p")
    adapter = BitgetLiveAdapter()

    calls: List[str] = []

    def _fake_request(method: str, path: str, **kwargs):
        _ = kwargs
        calls.append(f"{method}:{path}")
        if method == "GET" and path.startswith("/api/v2/spot/public/symbols"):
            return 200, {"code": "00000", "data": [{"minTradeAmount": "0.001", "quantityScale": "0.001"}]}
        if method == "POST" and path == "/api/v2/spot/trade/place-order":
            return 200, {"code": "00000", "data": {"orderId": "bg-oid-1"}}
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(adapter, "_request", _fake_request)

    child = {
        "client_order_id": "idem-bitget-1",
        "symbol": "BTC",
        "side": "buy",
        "qty": 0.1,
        "limit_price": 100.0,
        "tif": "IOC",
    }
    venue_order_id_1, ack_1 = adapter.submit_order(child, {"market_type": "spot"})
    before = len(calls)
    venue_order_id_2, ack_2 = adapter.submit_order(child, {"market_type": "spot"})

    assert venue_order_id_1 == "bg-oid-1"
    assert venue_order_id_2 == "bg-oid-1"
    assert before == len(calls)
    assert ack_1.get("status") == "submitted"
    assert ack_2.get("idempotency_hit") is True
