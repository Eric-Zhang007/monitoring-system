from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "monitoring"))

import paper_trading_daemon as daemon_mod  # noqa: E402


def test_run_cycle_writes_execution_event_chain(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    history_path = tmp_path / "history.jsonl"
    event_path = tmp_path / "events.jsonl"
    control_path = tmp_path / "control.json"

    def _fake_load_json(path: Path):
        if str(path).endswith("control.json"):
            return {"paper_enabled": True, "live_enabled": False}
        return {}

    def _fake_call_json(_sess, _method, url, *, payload=None, timeout_sec=12.0):
        _ = (payload, timeout_sec)
        if url.endswith("/signals/generate"):
            return {"signal_id": 11, "action": "buy", "confidence": 0.9, "strategy_bucket": "trend"}
        if url.endswith("/predict/liquid"):
            return {"outputs": {"current_price": 100.0}}
        if url.endswith("/execution/orders"):
            return {"decision_id": "d-1", "order_ids": [101]}
        if url.endswith("/execution/run"):
            return {
                "filled": 1,
                "rejected": 0,
                "reject_breakdown": {},
                "orders": [
                    {
                        "target": "BTC",
                        "metadata": {"horizon": "1d", "strategy_bucket": "trend"},
                        "execution_policy": "long_horizon_exec",
                        "execution_trace": {
                            "theoretical_price": 100.0,
                            "avg_fill_price": 100.2,
                            "filled_qty": 1.0,
                            "slippage_bps": 20.0,
                            "fees_paid": 0.1,
                            "impact_bps": 17.0,
                        },
                        "execution": {"status": "filled"},
                    }
                ],
            }
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(daemon_mod, "_http_session", lambda: object())
    monkeypatch.setattr(daemon_mod, "_load_json", _fake_load_json)
    monkeypatch.setattr(daemon_mod, "_call_json", _fake_call_json)

    out = daemon_mod._run_cycle(
        api_base="http://x",
        symbols=["BTC"],
        horizon="1d",
        min_confidence=0.5,
        strategy_id="s1",
        capital_per_order_usd=100.0,
        max_orders=8,
        timeout_sec=3.0,
        state_path=state_path,
        history_path=history_path,
        execution_event_path=event_path,
        control_path=control_path,
    )
    assert out["status"] == "ok"
    lines = [ln for ln in event_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["execution_policy"] == "long_horizon_exec"
    assert payload["horizon"] == "1d"
