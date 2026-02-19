from __future__ import annotations

from pathlib import Path
import sys

import pytest

pytest.importorskip("psycopg2")

sys.path.append(str(Path(__file__).resolve().parents[2] / "monitoring"))

import paper_trading_daemon as daemon_mod  # noqa: E402


def test_target_position_respects_neutral_deadband(monkeypatch):
    monkeypatch.setenv("PAPER_NEUTRAL_EPS", "0.001")
    out = daemon_mod._target_position({"expected_return": 0.0005, "signal_confidence": 1.0})
    assert out == 0.0


def test_target_position_direction_follows_signal_sign(monkeypatch):
    monkeypatch.setenv("PAPER_NEUTRAL_EPS", "0.0")
    long_pos = daemon_mod._target_position({"expected_return": 0.02, "signal_confidence": 0.8})
    short_pos = daemon_mod._target_position({"expected_return": -0.02, "signal_confidence": 0.8})
    assert long_pos > 0.0
    assert short_pos < 0.0
