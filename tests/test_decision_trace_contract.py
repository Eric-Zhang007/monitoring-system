from __future__ import annotations

from decision_trace.models import DecisionTrace


def test_decision_trace_contract_fields():
    trace = DecisionTrace(
        decision_id="d1",
        symbol="BTC",
        action="buy",
        target_pos=0.2,
        delta_pos=0.1,
        mu={"1h": 0.01},
        sigma={"1h": 0.02},
        cost={"1h": 0.001},
        risk={"regime": "GREEN"},
        account={"equity": 10000.0},
        reason_codes=["ok"],
    )
    payload = trace.model_dump()
    for key in ("decision_id", "symbol", "action", "target_pos", "delta_pos", "mu", "sigma", "cost", "risk", "account", "reason_codes"):
        assert key in payload
    assert payload["action"] in {"buy", "sell", "hold"}


def test_decision_trace_serialization_roundtrip():
    trace = DecisionTrace(
        decision_id="d2",
        symbol="ETH",
        action="hold",
        target_pos=0.0,
        delta_pos=0.0,
        exec_style="passive_twap",
        deadline_s=20,
        slices=8,
        reason_codes=["low_confidence"],
    )
    restored = DecisionTrace.model_validate(trace.model_dump(mode="json"))
    assert restored.exec_style == "passive_twap"
    assert restored.slices == 8
