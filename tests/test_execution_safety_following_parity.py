from __future__ import annotations

from pathlib import Path

from backend.execution_safety import (
    ExecutionSafetyController,
    KillSwitchAction,
    StopLossManager,
    read_kill_switch_action,
)


def test_plan_order_unsupported_switches_to_local_guard_and_safe_mode():
    ctrl = ExecutionSafetyController()
    ctrl.stoploss_mode = "trigger"
    out = ctrl.apply_startup_probe(plan_order_supported=False, safe_mode_on_failure=True)
    assert out["supported"] is False
    assert out["stoploss_mode"] == "local_guard"
    assert out["safe_mode"] is True


def test_ws_to_rest_with_local_guard_enters_safe_mode():
    ctrl = ExecutionSafetyController()
    ctrl.stoploss_mode = "local_guard"
    out = ctrl.apply_price_feed_status(ws_ok=False, rest_ok=True, rest_fallback_action_when_local_guard="safe_mode")
    assert out["price_feed_mode"] == "rest"
    assert out["safe_mode"] is True
    assert "ws_to_rest_with_local_guard" in str(out.get("reason") or "")


def test_local_guard_trigger_calls_protective_close():
    called = {}

    def _supports() -> bool:
        return False

    def _place(symbol: str, side: str, trigger: float, size: float):  # noqa: ARG001
        return {"ok": False, "reason": "unsupported"}

    def _close(symbol: str, side: str, size: float, reason: str):
        called["payload"] = {"symbol": symbol, "side": side, "size": size, "reason": reason}

    mgr = StopLossManager(supports_exchange_trigger=_supports, place_exchange_stop=_place, protective_close=_close)
    out = mgr.ensure_stop_loss(symbol="BTC", side="long", desired_sl_price=99.0, desired_size=1.5, mode="trigger")
    assert out["mode"] == "local_guard"
    fired = mgr.process_local_guards(mark_prices={"BTC": 98.0})
    assert len(fired) == 1
    assert called["payload"]["symbol"] == "BTC"
    assert called["payload"]["reason"] == "local_guard_triggered"
    assert mgr.safe_mode is True


def test_kill_switch_file_env_sqlite_channels(tmp_path: Path, monkeypatch):
    p = tmp_path / "KILL_SWITCH"
    p.write_text("panic_close", encoding="utf-8")
    act = read_kill_switch_action(file_path=str(p), env_key="TRADER_KILL_SWITCH_TEST")
    assert act == KillSwitchAction.PANIC_CLOSE

    p.unlink()
    monkeypatch.setenv("TRADER_KILL_SWITCH_TEST", "safe_mode")
    act_env = read_kill_switch_action(file_path=str(p), env_key="TRADER_KILL_SWITCH_TEST")
    assert act_env == KillSwitchAction.SAFE_MODE

    monkeypatch.delenv("TRADER_KILL_SWITCH_TEST", raising=False)
    act_sqlite = read_kill_switch_action(file_path=str(p), env_key="TRADER_KILL_SWITCH_TEST", sqlite_value="panic")
    assert act_sqlite == KillSwitchAction.PANIC_CLOSE


def test_unknown_position_triggers_safe_mode():
    ctrl = ExecutionSafetyController()
    out = ctrl.apply_unknown_positions(
        known_symbols=["BTC", "ETH"],
        exchange_positions=[{"symbol": "SOL", "size": 0.5}, {"symbol": "BTC", "size": 0.1}],
    )
    assert out["safe_mode"] is True
    assert out["unknown_symbols"] == ["SOL"]

