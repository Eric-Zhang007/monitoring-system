from __future__ import annotations

import json
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


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


data_mod.DataLoader = _Dummy
data_mod.TensorDataset = _Dummy
sys.modules["torch.utils"] = utils_mod
sys.modules["torch.utils.data"] = data_mod

import backbone_experiments as bb_mod  # noqa: E402


def _synthetic_rows(n: int = 900):
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        out.append(
            {
                "symbol": "BTC",
                "as_of_ts": base + timedelta(minutes=5 * i),
                "features": {
                    "ret_1": float(np.sin(i / 7.0) * 0.01),
                    "vol_12": float(0.1 + (i % 20) * 0.001),
                },
            }
        )
    return out


def test_backbone_experiments_main_writes_artifact_for_tft_with_torch_missing(monkeypatch, tmp_path):
    out_path = tmp_path / "backbone_suite_latest.json"
    monkeypatch.setattr(bb_mod, "LIQUID_FEATURE_KEYS", ["ret_1", "vol_12"])
    monkeypatch.setattr(bb_mod, "_load_rows", lambda *args, **kwargs: _synthetic_rows())
    monkeypatch.setattr(
        bb_mod,
        "_fit_predict_torch",
        lambda *args, **kwargs: (np.zeros((args[3].shape[0],), dtype=np.float64), "torch_missing"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backbone_experiments.py",
            "--database-url",
            "postgresql://unused",
            "--start",
            "2025-01-01T00:00:00Z",
            "--end",
            "2025-12-31T00:00:00Z",
            "--symbols",
            "BTC",
            "--lookback-steps",
            "8",
            "--horizon-steps",
            "1",
            "--max-samples",
            "700",
            "--backbones",
            "tft",
            "--out",
            str(out_path),
        ],
    )
    rc = bb_mod.main()
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["results"][0]["backbone"] == "tft"
    assert payload["results"][0]["walk_forward"]["status"] == "blocked"
    assert payload["results"][0]["walk_forward"]["reason"] == "torch_missing"
