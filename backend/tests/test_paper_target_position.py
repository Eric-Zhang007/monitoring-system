from __future__ import annotations

from pathlib import Path
import sys

import pytest

pytest.importorskip("requests")

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "monitoring"))

import paper_trading_daemon as daemon_mod  # noqa: E402


def test_target_position_zero_signal_forces_flat(monkeypatch):
    monkeypatch.setenv("PAPER_NEUTRAL_EPS", "0.000001")
    assert daemon_mod._target_position({"expected_return": 0.0, "signal_confidence": 1.0}) == 0.0


def test_target_position_small_signal_inside_deadband_forces_flat(monkeypatch):
    monkeypatch.setenv("PAPER_NEUTRAL_EPS", "0.001")
    assert daemon_mod._target_position({"expected_return": 0.0003, "signal_confidence": 0.9}) == 0.0
    assert daemon_mod._target_position({"expected_return": -0.0003, "signal_confidence": 0.9}) == 0.0


def test_target_position_large_signal_is_direction_symmetric(monkeypatch):
    monkeypatch.setenv("PAPER_NEUTRAL_EPS", "0")
    pos = daemon_mod._target_position({"expected_return": 0.02, "signal_confidence": 0.8})
    neg = daemon_mod._target_position({"expected_return": -0.02, "signal_confidence": 0.8})
    assert pos > 0.0
    assert neg < 0.0
    assert abs(pos + neg) < 1e-12
