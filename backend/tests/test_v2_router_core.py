from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from v2_router import _evaluate_risk, _ks_statistic, _psi, _walk_forward_metrics  # noqa: E402
from schemas_v2 import ExecuteOrdersRequest, RebalancePosition, SignalGenerateRequest  # noqa: E402
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
        RebalancePosition(target="AAPL", track="liquid", weight=0.3, sector="tech", style_bucket="growth"),
        RebalancePosition(target="NVDA", track="liquid", weight=0.3, sector="tech", style_bucket="growth"),
    ]
    adjusted, violations, gross, turnover = _evaluate_risk(
        proposed,
        [],
        0.0,
        max_sector_exposure_override=0.4,
        max_style_exposure_override=0.35,
    )
    assert adjusted
    assert any(v.startswith("sector_exposure_exceeded:tech") for v in violations)
    assert any(v.startswith("style_exposure_exceeded:growth") for v in violations)
    sector_exp = sum(abs(p.weight) for p in adjusted if (p.sector or "").lower() == "tech")
    style_exp = sum(abs(p.weight) for p in adjusted if (p.style_bucket or "").lower() == "growth")
    assert sector_exp <= 0.4 + 1e-9
    assert style_exp <= 0.35 + 1e-9
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
