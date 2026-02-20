from __future__ import annotations

from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

sys.path.append(str(Path(__file__).resolve().parents[1]))

from schemas_v2 import ExecuteOrdersRequest, ExecutionOrderInput, SubmitExecutionOrdersRequest


def test_execute_orders_request_accepts_coinbase_spot_without_product_type():
    req = ExecuteOrdersRequest(
        decision_id="d-1",
        adapter="coinbase_live",
        venue="coinbase",
        market_type="spot",
    )
    assert req.product_type is None


def test_execute_orders_request_accepts_bitget_spot_without_product_type():
    req = ExecuteOrdersRequest(
        decision_id="d-2",
        adapter="bitget_live",
        venue="bitget",
        market_type="spot",
    )
    assert req.product_type is None


def test_execute_orders_request_requires_product_type_for_bitget_perp():
    with pytest.raises(ValidationError, match="product_type is required"):
        ExecuteOrdersRequest(
            decision_id="d-3",
            adapter="bitget_live",
            venue="bitget",
            market_type="perp_usdt",
        )


def test_submit_execution_orders_request_validation_matrix():
    spot_req = SubmitExecutionOrdersRequest(
        adapter="bitget_live",
        venue="bitget",
        market_type="spot",
        orders=[ExecutionOrderInput(target="BTC", side="buy", quantity=0.01)],
    )
    assert spot_req.product_type is None

    perp_req = SubmitExecutionOrdersRequest(
        adapter="bitget_live",
        venue="bitget",
        market_type="perp_usdt",
        product_type="USDT-FUTURES",
        orders=[ExecutionOrderInput(target="BTCUSDT", side="buy", quantity=0.01)],
    )
    assert perp_req.product_type == "USDT-FUTURES"

    with pytest.raises(ValidationError, match="product_type is required"):
        SubmitExecutionOrdersRequest(
            adapter="bitget_live",
            venue="bitget",
            market_type="perp_usdt",
            orders=[ExecutionOrderInput(target="BTCUSDT", side="buy", quantity=0.01)],
        )
