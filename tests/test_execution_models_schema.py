from __future__ import annotations

from datetime import datetime, timezone

from execution_models import ChildOrder, ExecutionDecision, ExecutionLifecycleEvent, ExecutionOrder, Fill


def test_execution_models_fields_complete():
    now = datetime.now(timezone.utc)
    decision = ExecutionDecision(
        decision_id="d1",
        adapter="paper",
        venue="coinbase",
        market_type="spot",
        created_at=now,
        requested_by="api",
        policy="short_horizon_exec",
        risk_profile="balanced",
        cost_profile="standard",
    )
    assert decision.status == "created"

    order = ExecutionOrder(
        order_id=1,
        decision_id="d1",
        target="BTC",
        track="liquid",
        side="buy",
        quantity=0.1,
        est_price=100.0,
        adapter="paper",
        venue="coinbase",
        time_in_force="IOC",
        strategy_id="s1",
    )
    assert order.status == "submitted"

    child = ChildOrder(
        decision_id="d1",
        parent_order_id=1,
        client_order_id="c1",
        symbol="BTC",
        side="buy",
        qty=0.05,
        tif="IOC",
        slice_index=0,
    )
    assert child.status == "new"

    fill = Fill(child_order_id=1, fill_ts=now, qty=0.05, price=100.1, fee=0.01)
    assert fill.fee >= 0.0

    lifecycle = ExecutionLifecycleEvent(event="submit", status="ok", time=now.isoformat(), metrics={"latency_ms": 10.0})
    assert "latency_ms" in lifecycle.metrics
