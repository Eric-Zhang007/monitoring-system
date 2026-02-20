from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "inference"))

from model_router import ModelRouter  # noqa: E402


def test_tabular_residual_gate_prediction_uses_gate_and_delta(monkeypatch):
    router = ModelRouter()
    bundle = {
        "fusion_mode": "residual_gate",
        "feature_dim": 4,
        "x_mean": np.zeros(4, dtype=np.float32),
        "x_std": np.ones(4, dtype=np.float32),
        "base_feature_indices": np.array([0, 1], dtype=np.int64),
        "text_feature_indices": np.array([2, 3], dtype=np.int64),
        "base_weights": np.array([1.0, -1.0], dtype=np.float32),
        "text_weights": np.array([0.5, 0.5], dtype=np.float32),
        "gate_mu": 0.0,
        "gate_sigma": 1.0,
        "gate_multiplier": 1.0,
    }
    monkeypatch.setattr(router, "_load_liquid_tabular_model", lambda _symbol: bundle)

    pred, meta = router._predict_liquid_tabular_with_meta("BTC", np.array([2.0, 1.0, 4.0, 0.0], dtype=np.float32))
    # base=1.0, delta=2.0, gate=sigmoid(2.0)=0.880797...
    assert abs(float(pred) - 2.761594) < 1e-5
    assert str(meta.get("mode")) == "residual_gate"
    assert abs(float(meta.get("gate") or 0.0) - 0.880797) < 1e-5


def test_tabular_single_ridge_prediction_remains_compatible(monkeypatch):
    router = ModelRouter()
    bundle = {
        "fusion_mode": "single_ridge",
        "feature_dim": 3,
        "x_mean": np.zeros(3, dtype=np.float32),
        "x_std": np.ones(3, dtype=np.float32),
        "weights": np.array([0.2, 0.1, -0.5], dtype=np.float32),
    }
    monkeypatch.setattr(router, "_load_liquid_tabular_model", lambda _symbol: bundle)

    pred, meta = router._predict_liquid_tabular_with_meta("ETH", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert abs(float(pred) - (-1.1)) < 1e-6
    assert str(meta.get("mode")) == "single_ridge"


def test_predict_liquid_stack_exposes_tabular_fusion_fields(monkeypatch):
    router = ModelRouter()

    def _stub_tab(_symbol: str, _features: np.ndarray):
        return 0.12, {
            "mode": "residual_gate",
            "base": 0.07,
            "delta": 0.10,
            "gate": 0.5,
            "gate_multiplier": 1.1,
        }

    monkeypatch.setattr(router, "_predict_liquid_tabular_with_meta", _stub_tab)

    out = router.predict_liquid("SOL", np.array([0.0, 0.0], dtype=np.float32), model_name="liquid_baseline")
    stack = out.get("stack") if isinstance(out.get("stack"), dict) else {}
    assert abs(float(out.get("expected_return") or 0.0) - 0.12) < 1e-12
    assert str(stack.get("tabular_mode")) == "residual_gate"
    assert abs(float(stack.get("tabular_base") or 0.0) - 0.07) < 1e-12
    assert abs(float(stack.get("tabular_gate") or 0.0) - 0.5) < 1e-12
