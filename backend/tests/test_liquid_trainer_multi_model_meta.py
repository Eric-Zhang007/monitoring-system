from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "training"))


class _DummyPipeline:
    pass


def test_fit_horizon_models_with_meta_outputs_expected_structure():
    try:
        mod = importlib.import_module("liquid_model_trainer")
    except Exception as exc:
        pytest.skip(f"liquid_model_trainer unavailable in this env: {exc}")
    LIQUID_HORIZONS = list(getattr(mod, "LIQUID_HORIZONS", []))
    LiquidModelTrainer = getattr(mod, "LiquidModelTrainer")
    if not LIQUID_HORIZONS:
        pytest.skip("no LIQUID_HORIZONS")

    trainer = LiquidModelTrainer(pipeline=_DummyPipeline(), symbols=[])
    rng = np.random.default_rng(11)
    n = 80
    d = 10
    Xn = rng.normal(size=(n, d)).astype(np.float32)
    train_idx = np.arange(0, 64, dtype=np.int64)
    val_idx = np.arange(64, n, dtype=np.int64)
    y_by_h = {
        "1h": (0.12 * Xn[:, 0] - 0.05 * Xn[:, 1]).astype(np.float32),
        "4h": (0.10 * Xn[:, 1] + 0.03 * Xn[:, 2]).astype(np.float32),
        "1d": (0.08 * Xn[:, 2] - 0.02 * Xn[:, 3]).astype(np.float32),
        "7d": (0.06 * Xn[:, 4] + 0.01 * Xn[:, 5]).astype(np.float32),
    }
    vol = {h: np.maximum(1e-3, np.abs(v)).astype(np.float32) for h, v in y_by_h.items()}

    out = trainer._fit_horizon_models_with_meta(
        Xn_train_drop=Xn[train_idx],
        Xn_full=Xn,
        train_idx=train_idx,
        val_idx=val_idx,
        y_by_horizon=y_by_h,
        vol_targets=vol,
    )
    h_models = out.get("horizon_models") if isinstance(out.get("horizon_models"), dict) else {}
    meta = out.get("meta_aggregator") if isinstance(out.get("meta_aggregator"), dict) else {}
    assert set(h_models.keys()) == set(LIQUID_HORIZONS)
    assert list(meta.get("horizons") or []) == LIQUID_HORIZONS
    assert len(list(meta.get("weights") or [])) == len(LIQUID_HORIZONS)
    metrics = meta.get("metrics") if isinstance(meta.get("metrics"), dict) else {}
    assert float(metrics.get("mse") or 0.0) >= 0.0
