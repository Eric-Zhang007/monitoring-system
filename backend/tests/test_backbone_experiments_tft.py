from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "training"))

torch_stub = types.ModuleType("torch")
torch_stub.manual_seed = lambda *args, **kwargs: None
torch_stub.cuda = types.SimpleNamespace(
    is_available=lambda: False,
    manual_seed_all=lambda *args, **kwargs: None,
)
torch_stub.device = lambda name: name
sys.modules["torch"] = torch_stub

nn_mod = types.ModuleType("torch.nn")


class _Module:
    pass


nn_mod.Module = _Module
sys.modules["torch.nn"] = nn_mod
torch_stub.nn = nn_mod

utils_mod = types.ModuleType("torch.utils")
data_mod = types.ModuleType("torch.utils.data")


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


data_mod.DataLoader = _Dummy
data_mod.TensorDataset = _Dummy
sys.modules["torch.utils"] = utils_mod
sys.modules["torch.utils.data"] = data_mod

import backbone_experiments as bb_mod  # noqa: E402


def test_fold_metrics_routes_tft_to_torch_branch(monkeypatch):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4, 3)).astype(np.float32)
    y = rng.normal(size=(80,)).astype(np.float32)
    folds = [(np.arange(0, 60), np.arange(60, 80))]
    seen = {"called": False}

    def _fake_fit(backbone, X_train, y_train, X_test, **kwargs):
        seen["called"] = True
        assert backbone == "tft"
        return np.zeros((X_test.shape[0],), dtype=np.float64), "ok"

    monkeypatch.setattr(bb_mod, "_fit_predict_torch", _fake_fit)
    out = bb_mod._fold_metrics(
        X,
        y,
        folds,
        backbone="tft",
        l2=0.05,
        seed=7,
        epochs=1,
        batch_size=16,
        lr=1e-3,
        min_train_points=16,
        min_test_points=8,
    )
    assert seen["called"] is True
    assert out["status"] == "ok"
    assert int(out["folds"]) == 1


def test_fold_metrics_marks_tft_blocked_when_torch_missing(monkeypatch):
    X = np.zeros((80, 4, 3), dtype=np.float32)
    y = np.zeros((80,), dtype=np.float32)
    folds = [(np.arange(0, 60), np.arange(60, 80))]
    monkeypatch.setattr(
        bb_mod,
        "_fit_predict_torch",
        lambda *args, **kwargs: (np.zeros((20,), dtype=np.float64), "torch_missing"),
    )
    out = bb_mod._fold_metrics(
        X,
        y,
        folds,
        backbone="tft",
        l2=0.05,
        seed=7,
        epochs=1,
        batch_size=16,
        lr=1e-3,
        min_train_points=16,
        min_test_points=8,
    )
    assert out["status"] == "blocked"
    assert str(out["reason"]) == "torch_missing"
