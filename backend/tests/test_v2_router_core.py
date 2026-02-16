from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402
from v2_router import _evaluate_risk, _execution_volatility_violations, _infer_daily_loss_ratio, _ks_statistic, _normalize_execution_payload, _psi, _walk_forward_metrics  # noqa: E402
from schemas_v2 import BacktestRunRequest, ExecuteOrdersRequest, RebalancePosition, RiskCheckRequest, SignalGenerateRequest  # noqa: E402
from execution_engine import ExecutionEngine  # noqa: E402


def test_evaluate_risk_caps_single_weight():
    proposed = [RebalancePosition(target="BTC", track="liquid", weight=0.9)]
    current = []
    adjusted, violations, gross, turnover = _evaluate_risk(proposed, current, 0.0)
    assert adjusted
    assert abs(adjusted[0].weight) <= 0.2 + 1e-9
    assert "single_weight_exceeded:BTC" in violations
    assert gross <= 1.0 + 1e-9
    assert turnover >= 0.0


def test_evaluate_risk_caps_sector_and_style_exposure():
    proposed = [
        RebalancePosition(target="AAPL", track="liquid", weight=0.2, sector="tech", style_bucket="growth"),
        RebalancePosition(target="NVDA", track="liquid", weight=0.2, sector="tech", style_bucket="growth"),
    ]
    adjusted, violations, gross, turnover = _evaluate_risk(
        proposed,
        [],
        0.0,
        max_sector_exposure_override=0.3,
        max_style_exposure_override=0.25,
    )
    assert adjusted
    assert any(v.startswith("sector_exposure_exceeded:tech") for v in violations)
    assert any(v.startswith("style_exposure_exceeded:growth") for v in violations)
    sector_exp = sum(abs(p.weight) for p in adjusted if (p.sector or "").lower() == "tech")
    style_exp = sum(abs(p.weight) for p in adjusted if (p.style_bucket or "").lower() == "growth")
    assert sector_exp <= 0.3 + 1e-9
    assert style_exp <= 0.25 + 1e-9
    assert gross <= 1.0 + 1e-9
    assert turnover >= 0.0


def test_walk_forward_metrics_outputs():
    now = datetime.now(timezone.utc)
    rows = []
    p = 100.0
    for i in range(80):
        p = p * (1.0 + (0.001 if i % 2 == 0 else -0.0006))
        rows.append({"price": p, "timestamp": now - timedelta(days=80 - i)})
    m = _walk_forward_metrics(rows, train_days=20, test_days=5, fee_bps=5, slippage_bps=3)
    assert "ic" in m
    assert "hit_rate" in m
    assert "pnl_after_cost" in m


def test_walk_forward_metrics_keeps_weekend_rows_for_crypto():
    start = datetime(2026, 2, 14, 0, 0, tzinfo=timezone.utc)  # Saturday
    rows = []
    p = 100.0
    for i in range(40):
        p = p * (1.0 + (0.001 if i % 2 == 0 else -0.0007))
        rows.append({"price": p, "volume": 1000.0 + i, "timestamp": start + timedelta(hours=i)})
    m = _walk_forward_metrics(rows, train_days=20, test_days=5, fee_bps=5, slippage_bps=3)
    assert m["samples"] > 0


def test_execution_engine_paper():
    engine = ExecutionEngine()
    orders = [{"quantity": 1.0, "side": "buy", "est_price": 100.0}]
    out = engine.run("paper", orders)
    assert len(out) == 1
    assert out[0]["status"] in {"filled", "rejected"}


def test_drift_stats_helpers():
    ref = [0.01, 0.02, 0.0, -0.01, 0.015, 0.011]
    cur = [0.03, 0.025, 0.01, 0.02, 0.018, 0.028]
    psi = _psi(ref, cur)
    ks = _ks_statistic(ref, cur)
    assert psi >= 0.0
    assert 0.0 <= ks <= 1.0


def test_extended_request_models():
    sig = SignalGenerateRequest(track="liquid", target="BTC", strategy_id="s1", cost_profile="std", risk_profile="safe")
    assert sig.strategy_id == "s1"
    run = ExecuteOrdersRequest(
        decision_id="d1",
        adapter="paper",
        time_in_force="IOC",
        max_slippage_bps=10.0,
        venue="coinbase",
    )
    assert run.venue == "coinbase"
    run_bitget = ExecuteOrdersRequest(
        decision_id="d2",
        adapter="bitget_live",
        venue="bitget",
        market_type="perp_usdt",
        product_type="USDT-FUTURES",
        leverage=3.0,
        reduce_only=False,
        position_mode="one_way",
        margin_mode="cross",
    )
    assert run_bitget.adapter == "bitget_live"
    bt = BacktestRunRequest(track="liquid", targets=["BTC"])
    assert bt.run_source == "prod"
    assert bt.alignment_mode == "strict_asof"
    assert bt.alignment_version == "strict_asof_v1"
    assert bt.fee_bps == 5.0
    assert bt.slippage_bps == 3.0


def test_default_model_dir_prefers_repo_path_in_nodocker(monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    path = router_mod._default_model_dir()
    assert path.name == "models"
    assert "backend" in str(path).replace("\\", "/")


def test_risk_check_kill_switch_violation_code_is_consistent(monkeypatch):
    class _FakeRepo:
        def __init__(self):
            self.events = []

        def is_kill_switch_triggered(self, track: str, strategy_id: str) -> bool:
            return track == "liquid" and strategy_id == "global"

        def upsert_kill_switch_state(self, *args, **kwargs):
            return {}

        def save_risk_event(self, **kwargs):
            self.events.append(kwargs)

    fake_repo = _FakeRepo()
    monkeypatch.setattr(router_mod, "repo", fake_repo)
    payload = RiskCheckRequest(
        proposed_positions=[RebalancePosition(target="BTC", track="liquid", weight=0.1)],
        current_positions=[],
        realized_drawdown=0.0,
    )
    resp = asyncio.run(router_mod.risk_check(payload))
    assert resp.approved is False
    assert "kill_switch_triggered:liquid:global" in resp.violations


def test_risk_check_runtime_limits_trigger_violation(monkeypatch):
    class _FakeRepo:
        def __init__(self):
            self.last_upsert = {}

        def is_kill_switch_triggered(self, track: str, strategy_id: str) -> bool:
            return False

        def upsert_kill_switch_state(self, *args, **kwargs):
            self.last_upsert = kwargs
            return {}

        def save_risk_event(self, **kwargs):
            return None

    monkeypatch.setenv("RISK_MAX_DAILY_LOSS", "0.01")
    monkeypatch.setenv("RISK_MAX_CONSECUTIVE_LOSSES", "2")
    fake = _FakeRepo()
    monkeypatch.setattr(router_mod, "repo", fake)
    payload = RiskCheckRequest(
        proposed_positions=[RebalancePosition(target="BTC", track="liquid", weight=0.1)],
        current_positions=[],
        realized_drawdown=0.0,
        daily_loss=0.02,
        consecutive_losses=3,
    )
    resp = asyncio.run(router_mod.risk_check(payload))
    assert resp.approved is False
    assert "daily_loss_exceeded" in resp.violations
    assert "consecutive_loss_exceeded" in resp.violations
    assert fake.last_upsert.get("reason") == "daily_loss_exceeded"
    assert int(fake.last_upsert.get("duration_minutes") or 0) >= 1


def test_risk_check_stop_loss_and_intraday_halt(monkeypatch):
    class _FakeRepo:
        def __init__(self):
            self.last_upsert = {}

        def is_kill_switch_triggered(self, track: str, strategy_id: str) -> bool:
            return False

        def upsert_kill_switch_state(self, *args, **kwargs):
            self.last_upsert = kwargs
            return {}

        def save_risk_event(self, **kwargs):
            return None

    monkeypatch.setenv("RISK_SINGLE_STOP_LOSS_PCT", "0.01")
    monkeypatch.setenv("RISK_INTRADAY_DRAWDOWN_HALT_PCT", "0.03")
    fake = _FakeRepo()
    monkeypatch.setattr(router_mod, "repo", fake)
    payload = RiskCheckRequest(
        proposed_positions=[RebalancePosition(target="BTC", track="liquid", weight=0.02)],
        current_positions=[],
        realized_drawdown=0.0,
        latest_trade_edge_ratio=-0.02,
        intraday_drawdown=0.05,
    )
    resp = asyncio.run(router_mod.risk_check(payload))
    assert resp.approved is False
    assert "single_trade_stop_loss_triggered" in resp.violations
    assert "intraday_drawdown_halt" in resp.violations
    assert fake.last_upsert.get("reason") == "single_trade_stop_loss_triggered"


def test_risk_check_drawdown_near_limit_enforces_reduce_only(monkeypatch):
    class _FakeRepo:
        def is_kill_switch_triggered(self, track: str, strategy_id: str) -> bool:
            return False

        def upsert_kill_switch_state(self, *args, **kwargs):
            return {}

        def save_risk_event(self, **kwargs):
            return None

    monkeypatch.setenv("RISK_MAX_DRAWDOWN", "0.12")
    monkeypatch.setenv("RISK_DRAWDOWN_WARN_THRESHOLD", "0.08")
    monkeypatch.setenv("RISK_DRAWDOWN_NEAR_LIMIT", "0.10")
    monkeypatch.setattr(router_mod, "repo", _FakeRepo())
    payload = RiskCheckRequest(
        proposed_positions=[RebalancePosition(target="BTC", track="liquid", weight=0.08)],
        current_positions=[RebalancePosition(target="BTC", track="liquid", weight=0.02)],
        realized_drawdown=0.101,
    )
    resp = asyncio.run(router_mod.risk_check(payload))
    assert resp.approved is False
    assert "drawdown_near_limit_reduce_only" in resp.violations
    assert abs(resp.adjusted_positions[0].weight - 0.02) < 1e-9


def test_infer_daily_loss_ratio_uses_notional_denominator(monkeypatch):
    class _FakeRepo:
        def build_pnl_attribution(self, track: str, lookback_hours: int):
            return {"totals": {"net_pnl": -100.0, "gross_notional_signed": 10000.0}}

    monkeypatch.setattr(router_mod, "repo", _FakeRepo())
    ratio = _infer_daily_loss_ratio("liquid")
    assert abs(ratio - 0.01) < 1e-9


def test_run_execution_blocks_on_abnormal_volatility(monkeypatch):
    class _FakeRepo:
        def is_kill_switch_triggered(self, track: str, strategy_id: str) -> bool:
            return False

        def fetch_orders_for_decision(self, decision_id: str, limit: int = 100):
            return [{"id": 1, "target": "BTC", "track": "liquid", "side": "buy", "quantity": 1.0, "metadata": {}}]

        def get_model_rollout_state(self, track: str):
            return {"stage_pct": 100}

        def load_price_history(self, symbol: str, lookback_days: int = 90):
            prices = [100.0 + i * 0.1 for i in range(40)]
            prices[-1] = prices[-2] * 1.2
            return [{"price": p} for p in prices]

        def upsert_kill_switch_state(self, **kwargs):
            return {}

        def save_risk_event(self, **kwargs):
            return None

        def get_execution_edge_pnls(self, track: str, lookback_hours: int, limit: int = 500, strategy_id: str | None = None):
            return []

    monkeypatch.setattr(router_mod, "repo", _FakeRepo())
    monkeypatch.setenv("RISK_MAX_ABS_RETURN", "0.05")
    req = ExecuteOrdersRequest(decision_id="d1", adapter="paper", time_in_force="IOC", max_slippage_bps=10.0, venue="coinbase")
    try:
        asyncio.run(router_mod.run_execution(req))
        assert False, "expected HTTPException"
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        detail = getattr(exc, "detail", "")
        assert status_code == 423
        assert "risk_blocked:abnormal_volatility_circuit_breaker:BTC" in str(detail)


def test_normalize_execution_payload_has_fixed_lifecycle_shape():
    raw = {
        "status": "filled",
        "filled_qty": 1.0,
        "lifecycle": [
            {"event": "limit_submit", "status": "accepted", "filled_qty": 0.4, "http_code": 200},
            {"event": "market_submit", "status": "filled", "filled_qty": 0.6, "retry": 1},
        ],
    }
    out = _normalize_execution_payload(raw)
    assert out["status"] == "filled"
    assert isinstance(out["lifecycle"], list)
    assert out["lifecycle"][0]["event"] == "limit_submit"
    assert out["lifecycle"][0]["status"] == "accepted"
    assert "time" in out["lifecycle"][0]
    assert isinstance(out["lifecycle"][0]["metrics"], dict)


def test_run_execution_blocks_on_strategy_consecutive_losses(monkeypatch):
    class _FakeRepo:
        def is_kill_switch_triggered(self, track: str, strategy_id: str) -> bool:
            return False

        def fetch_orders_for_decision(self, decision_id: str, limit: int = 100):
            return [{"id": 1, "target": "BTC", "track": "liquid", "side": "buy", "quantity": 1.0, "strategy_id": "strat-a", "metadata": {}}]

        def get_model_rollout_state(self, track: str):
            return {"stage_pct": 100}

        def load_price_history(self, symbol: str, lookback_days: int = 90):
            return [{"price": 100.0 + i * 0.01} for i in range(60)]

        def get_execution_consecutive_losses(self, track: str, lookback_hours: int = 24, limit: int = 200, strategy_id: str | None = None):
            return 3 if strategy_id == "strat-a" else 0

        def get_execution_edge_pnls(self, track: str, lookback_hours: int, limit: int = 500, strategy_id: str | None = None):
            return []

        def upsert_kill_switch_state(self, **kwargs):
            return {}

        def save_risk_event(self, **kwargs):
            return None

    monkeypatch.setattr(router_mod, "repo", _FakeRepo())
    monkeypatch.setenv("RISK_MAX_CONSECUTIVE_LOSSES", "2")
    req = ExecuteOrdersRequest(decision_id="d1", adapter="paper", time_in_force="IOC", max_slippage_bps=10.0, venue="coinbase")
    try:
        asyncio.run(router_mod.run_execution(req))
        assert False, "expected HTTPException"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 423
        assert "risk_blocked:consecutive_loss_exceeded:strat-a" in str(getattr(exc, "detail", ""))


def test_run_execution_blocks_on_take_profit_precheck(monkeypatch):
    class _FakeRepo:
        def is_kill_switch_triggered(self, track: str, strategy_id: str) -> bool:
            return False

        def fetch_orders_for_decision(self, decision_id: str, limit: int = 100):
            return [{"id": 1, "target": "BTC", "track": "liquid", "side": "buy", "quantity": 1.0, "strategy_id": "strat-a", "metadata": {}}]

        def get_model_rollout_state(self, track: str):
            return {"stage_pct": 100}

        def load_price_history(self, symbol: str, lookback_days: int = 90):
            return [{"price": 100.0 + i * 0.01} for i in range(80)]

        def get_execution_consecutive_losses(self, track: str, lookback_hours: int = 24, limit: int = 200, strategy_id: str | None = None):
            return 0

        def get_execution_edge_pnls(self, track: str, lookback_hours: int, limit: int = 500, strategy_id: str | None = None):
            return [{"created_at": datetime.now(timezone.utc), "edge_pnl": 5.0, "notional": 100.0}]

        def upsert_kill_switch_state(self, **kwargs):
            return {}

        def save_risk_event(self, **kwargs):
            return None

    monkeypatch.setenv("RISK_SINGLE_TAKE_PROFIT_PCT", "0.03")
    monkeypatch.setattr(router_mod, "repo", _FakeRepo())
    req = ExecuteOrdersRequest(decision_id="d1", adapter="paper", time_in_force="IOC", max_slippage_bps=10.0, venue="coinbase")
    try:
        asyncio.run(router_mod.run_execution(req))
        assert False, "expected HTTPException"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 423
        assert "risk_blocked:single_trade_take_profit_reached:strat-a" in str(getattr(exc, "detail", ""))


def test_opening_status_returns_remaining_seconds(monkeypatch):
    now = datetime.now(timezone.utc)

    class _FakeRepo:
        def get_kill_switch_state(self, track: str, strategy_id: str):
            return {"state": "triggered", "reason": "x", "updated_at": now, "expires_at": now + timedelta(seconds=30)}

        def is_kill_switch_triggered(self, track: str, strategy_id: str) -> bool:
            return True

    monkeypatch.setattr(router_mod, "repo", _FakeRepo())
    resp = asyncio.run(router_mod.get_opening_status(track="liquid", strategy_id="strat-a"))
    assert resp.can_open_new_positions is False
    assert resp.remaining_seconds > 0


def test_execution_volatility_uses_symbol_threshold_override(monkeypatch):
    class _FakeRepo:
        def load_price_history(self, symbol: str, lookback_days: int = 2):
            prices = [100.0 + i * 0.1 for i in range(40)]
            prices[-1] = prices[-2] * 1.12
            return [{"price": p} for p in prices]

    monkeypatch.setattr(router_mod, "repo", _FakeRepo())
    monkeypatch.setenv("RISK_MAX_ABS_RETURN", "0.05")
    monkeypatch.setenv("RISK_MAX_ABS_RETURN_SYMBOLS", "BTC=0.20")
    out = _execution_volatility_violations([{"target": "BTC"}])
    assert out == []
