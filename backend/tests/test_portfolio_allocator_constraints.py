from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "backend"))

from portfolio_allocator import AllocatorSignal, allocate_targets  # noqa: E402


def _fake_price_history(_target: str, lookback_days: int):
    _ = lookback_days
    return [{"price": 100.0 + i} for i in range(64)]


def test_allocator_respects_risk_budget_and_symbol_cap(monkeypatch):
    monkeypatch.setenv("ALLOCATOR_SINGLE_SYMBOL_MAX", "0.2")
    monkeypatch.setenv("ALLOCATOR_BUCKET_LIMITS", "trend=0.5,event=0.6,mean_reversion=0.4")
    sigs = [
        AllocatorSignal(target="BTC", track="liquid", horizon="1h", action="buy", score=0.6, confidence=0.9, strategy_bucket="trend"),
        AllocatorSignal(target="ETH", track="liquid", horizon="4h", action="buy", score=0.4, confidence=0.8, strategy_bucket="trend"),
        AllocatorSignal(target="SOL", track="liquid", horizon="1d", action="sell", score=0.5, confidence=0.7, strategy_bucket="event"),
    ]
    out = allocate_targets(signals=sigs, risk_budget=1.0, load_price_history=_fake_price_history)
    weights = out["weights"]
    assert weights
    gross = sum(abs(float(v)) for v in weights.values())
    assert gross <= 1.0 + 1e-9
    assert all(abs(float(v)) <= 0.2 + 1e-9 for v in weights.values())
