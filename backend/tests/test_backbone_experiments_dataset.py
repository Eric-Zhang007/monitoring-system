from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
dist_data_mod = types.ModuleType("torch.utils.data.distributed")


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


data_mod.DataLoader = _Dummy
data_mod.TensorDataset = _Dummy
dist_data_mod.DistributedSampler = _Dummy
sys.modules["torch.utils"] = utils_mod
sys.modules["torch.utils.data"] = data_mod
sys.modules["torch.utils.data.distributed"] = dist_data_mod

dist_mod = types.ModuleType("torch.distributed")
dist_mod.is_available = lambda: False
dist_mod.is_initialized = lambda: False
sys.modules["torch.distributed"] = dist_mod

from backbone_experiments import build_sequence_dataset  # noqa: E402
from backbone_experiments import normalize_backbones  # noqa: E402


def _rows(symbol: str, n: int):
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        out.append(
            {
                "symbol": symbol,
                "as_of_ts": base + timedelta(minutes=5 * i),
                "features": {
                    "ret_1": 0.01 if i % 2 == 0 else -0.01,
                    "vol_12": 0.1 + i * 0.001,
                },
            }
        )
    return out


def test_build_sequence_dataset_shape_and_target_alignment():
    rows = _rows("BTC", 12) + _rows("ETH", 10)
    X, y = build_sequence_dataset(
        rows,
        lookback_steps=4,
        horizon_steps=1,
        feature_keys=["ret_1", "vol_12"],
        max_samples=0,
    )
    # BTC: 12 - 4 - 1 + 1 = 8, ETH: 10 - 4 - 1 + 1 = 6
    assert X.shape == (14, 4, 2)
    assert y.shape == (14,)
    # First sample target should be row index 4 (0-based).
    assert abs(float(y[0]) - float(rows[4]["features"]["ret_1"])) < 1e-8


def test_build_sequence_dataset_applies_max_samples_deterministically():
    rows = _rows("BTC", 40)
    X, y = build_sequence_dataset(
        rows,
        lookback_steps=5,
        horizon_steps=1,
        feature_keys=["ret_1", "vol_12"],
        max_samples=7,
    )
    assert X.shape[0] == 7
    assert y.shape[0] == 7
    # Subsampling keeps ordering and reproducibility via linspace index selection.
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y))


def test_normalize_backbones_keeps_supported_and_applies_aliases():
    inp = ["ridge", "itransformer", "patchtst", "tftlite", "unknown", "tft", "RIDGE"]
    out = normalize_backbones(inp)
    assert out == ["ridge", "itransformer", "patchtst", "tft"]
