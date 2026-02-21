from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import v2_router as router_mod  # noqa: E402


def test_liquid_multi_horizon_signal_returns_edge_and_score_maps(monkeypatch):
    monkeypatch.setenv("SIGNAL_SCORE_ENTRY_BY_HORIZON", "1h=0.5,4h=0.4,1d=0.3,7d=0.2")
    monkeypatch.setenv("SIGNAL_CONFIDENCE_MIN_BY_HORIZON", "1h=0.5,4h=0.5,1d=0.5,7d=0.5")
    out = router_mod._liquid_multi_horizon_signal(
        pred_outputs={
            "expected_return_horizons": {"1h": 0.02, "4h": 0.03, "1d": 0.04, "7d": 0.05},
            "signal_confidence_horizons": {"1h": 0.8, "4h": 0.8, "1d": 0.8, "7d": 0.8},
            "vol_forecast_horizons": {"1h": 0.02, "4h": 0.03, "1d": 0.04, "7d": 0.05},
        },
        horizon="4h",
        cost_profile="standard",
    )
    assert out["selected_horizon"] == "4h"
    assert set(out["score_horizons"].keys()) == {"1h", "4h", "1d", "7d"}
    assert set(out["edge_horizons"].keys()) == {"1h", "4h", "1d", "7d"}
    assert out["action"] in {"buy", "sell", "hold"}
