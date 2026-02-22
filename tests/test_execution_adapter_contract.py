from __future__ import annotations

from execution_engine import ExecutionEngine


REQUIRED_METHODS = ("prepare", "submit_order", "poll_order", "cancel_order", "fetch_fills", "fetch_positions")


def test_all_adapters_implement_contract_methods():
    engine = ExecutionEngine()
    for name in ("paper", "coinbase_live", "bitget_live"):
        adapter = engine.adapters[name]
        for method in REQUIRED_METHODS:
            fn = getattr(adapter, method, None)
            assert callable(fn), f"{name} missing method: {method}"


def test_paper_adapter_submit_poll_fetch_fills_contract():
    engine = ExecutionEngine()
    adapter = engine.adapters["paper"]
    venue_order_id, ack = adapter.submit_order(
        {
            "client_order_id": "d1:1:0:0",
            "symbol": "BTC",
            "side": "buy",
            "qty": 0.25,
            "limit_price": 100.0,
            "tif": "IOC",
        },
        {"market_state": {"bid_px": 99.9, "ask_px": 100.1, "spread_bps": 20.0, "imbalance": 0.1}, "fee_bps": 5.0},
    )
    assert venue_order_id
    assert set(("status", "filled_qty", "avg_fill_price", "fees_paid")).issubset(set(ack.keys()))
    polled = adapter.poll_order(venue_order_id, timeout=1.0)
    assert set(("status", "filled_qty", "avg_fill_price", "fees_paid")).issubset(set(polled.keys()))
    fills = adapter.fetch_fills(venue_order_id)
    if fills:
        assert set(("qty", "price", "fee")).issubset(set(fills[0].keys()))


def test_coinbase_adapter_submit_poll_fetch_fills_contract(monkeypatch):
    engine = ExecutionEngine()
    adapter = engine.adapters["coinbase_live"]
    adapter.api_key_name = "k"
    adapter.api_private_key = "s"

    def _fake_request(method: str, path: str, **kwargs):
        _ = kwargs
        if method == "POST" and path == "/api/v3/brokerage/orders":
            return 200, {"success_response": {"order_id": "cb-1"}, "order": {"status": "OPEN"}}
        if method == "GET" and path.startswith("/api/v3/brokerage/orders/historical/cb-1"):
            return 200, {"order": {"status": "FILLED", "filled_size": "0.20", "average_filled_price": "101.0", "total_fees": "0.01"}}
        if method == "GET" and path == "/api/v3/brokerage/orders/historical/fills":
            return 200, {"fills": [{"size": "0.20", "price": "101.0", "commission": "0.01"}]}
        if method == "POST" and path == "/api/v3/brokerage/orders/batch_cancel":
            return 200, {}
        if method == "GET" and path == "/api/v3/brokerage/accounts":
            return 200, {"accounts": []}
        return 200, {}

    monkeypatch.setattr(adapter, "_request", _fake_request)
    venue_order_id, ack = adapter.submit_order(
        {
            "client_order_id": "d1:1:0:0",
            "symbol": "BTC",
            "side": "buy",
            "qty": 0.2,
            "limit_price": 101.0,
            "tif": "IOC",
        },
        {"time_in_force": "IOC"},
    )
    assert venue_order_id == "cb-1"
    assert set(("status", "filled_qty", "avg_fill_price", "fees_paid")).issubset(set(ack.keys()))
    polled = adapter.poll_order("cb-1", timeout=1.0)
    assert set(("status", "filled_qty", "avg_fill_price", "fees_paid")).issubset(set(polled.keys()))
    fills = adapter.fetch_fills("cb-1")
    assert fills
    assert set(("qty", "price", "fee")).issubset(set(fills[0].keys()))


def test_bitget_adapter_submit_poll_fetch_fills_contract(monkeypatch):
    engine = ExecutionEngine()
    adapter = engine.adapters["bitget_live"]
    adapter.api_key = "k"
    adapter.api_secret = "s"
    adapter.api_passphrase = "p"

    def _fake_request(method: str, path: str, **kwargs):
        _ = kwargs
        if method == "GET" and path.startswith("/api/v2/spot/public/symbols"):
            return 200, {"code": "00000", "data": [{"minTradeAmount": "0.001", "quantityScale": "0.001"}]}
        if method == "POST" and path == "/api/v2/spot/trade/place-order":
            return 200, {"code": "00000", "data": {"orderId": "bg-1"}}
        if method == "POST" and path == "/api/v2/spot/trade/orderInfo":
            return 200, {"code": "00000", "data": {"status": "filled", "filledQty": "0.1", "avgPrice": "100.5"}}
        if method == "POST" and path == "/api/v2/spot/trade/fills":
            return 200, {"code": "00000", "data": [{"fillQty": "0.1", "price": "100.5", "fee": "0.001"}]}
        if method == "POST" and path == "/api/v2/spot/trade/cancel-order":
            return 200, {"code": "00000", "data": {}}
        if method == "GET" and path == "/api/v2/spot/account/assets":
            return 200, {"code": "00000", "data": []}
        if method == "POST" and path == "/api/v2/mix/position/all-position":
            return 200, {"code": "00000", "data": []}
        return 200, {"code": "00000", "data": {}}

    monkeypatch.setattr(adapter, "_request", _fake_request)
    venue_order_id, ack = adapter.submit_order(
        {
            "client_order_id": "d1:1:0:0",
            "symbol": "BTC",
            "side": "buy",
            "qty": 0.1,
            "limit_price": 100.0,
            "tif": "IOC",
        },
        {"market_type": "spot"},
    )
    assert venue_order_id == "bg-1"
    assert set(("status", "filled_qty", "avg_fill_price", "fees_paid")).issubset(set(ack.keys()))
    polled = adapter.poll_order("bg-1", timeout=1.0)
    assert set(("status", "filled_qty", "avg_fill_price", "fees_paid")).issubset(set(polled.keys()))
    fills = adapter.fetch_fills("bg-1")
    assert fills
    assert set(("qty", "price", "fee")).issubset(set(fills[0].keys()))
