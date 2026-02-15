from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from execution_engine import ExecutionEngine, PaperExecutionAdapter


def test_paper_execution_reject_path_has_lifecycle():
    adapter = PaperExecutionAdapter(reject_rate=1.0, partial_fill_rate=0.0)
    out = adapter.execute({"quantity": 1.0, "side": "buy", "est_price": 100.0}, context={"max_retries": 1})
    assert out["status"] == "rejected"
    assert isinstance(out.get("lifecycle"), list)
    assert out["lifecycle"]
    assert out["lifecycle"][0]["event"] == "limit_submit"


def test_paper_execution_partial_fill_path_has_cancel_and_market_events():
    adapter = PaperExecutionAdapter(reject_rate=0.0, partial_fill_rate=1.0)
    out = adapter.execute({"quantity": 2.0, "side": "buy", "est_price": 100.0}, context={"max_retries": 2})
    assert out["status"] in {"filled", "partially_filled"}
    lifecycle = out.get("lifecycle", [])
    events = [x.get("event") for x in lifecycle]
    assert "limit_submit" in events
    assert "cancel_limit" in events
    assert "market_submit" in events


def test_execution_engine_invalid_adapter_raises():
    engine = ExecutionEngine()
    try:
        engine.run("invalid", [{"quantity": 1.0, "side": "buy", "est_price": 1.0}])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unsupported adapter" in str(exc)
