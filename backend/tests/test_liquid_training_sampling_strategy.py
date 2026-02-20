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
torch_stub.backends = types.SimpleNamespace(
    cudnn=types.SimpleNamespace(deterministic=False, benchmark=False)
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

nn_parallel_mod = types.ModuleType("torch.nn.parallel")
nn_parallel_mod.DistributedDataParallel = _Dummy
sys.modules["torch.nn.parallel"] = nn_parallel_mod

from feature_pipeline import FeaturePipeline, SampleBatch  # noqa: E402
from liquid_model_trainer import LIQUID_FEATURE_KEYS, LiquidModelTrainer  # noqa: E402


def test_sampling_indices_are_deterministic_for_uniform_and_tail_modes():
    idx_uniform = FeaturePipeline._sampling_indices(10, 4, "uniform")
    idx_tail = FeaturePipeline._sampling_indices(10, 4, "tail")
    assert idx_uniform.tolist() == [0, 3, 6, 9]
    assert idx_tail.tolist() == [6, 7, 8, 9]


class _FakePipeline:
    def __init__(self):
        self.kwargs = {}

    def check_data_quality(self, symbol: str, timeframe: str, lookback_hours: int):
        return {
            "quality_passed": 1.0,
            "total_rows": 1000.0,
            "required_rows": 200.0,
            "missing_rate": 0.0,
            "invalid_price_rate": 0.0,
            "duplicate_rate": 0.0,
            "stale_ratio": 0.0,
        }

    def load_liquid_training_batch(self, **kwargs):
        self.kwargs = dict(kwargs)
        return SampleBatch(
            X=np.zeros((0, 4), dtype=np.float32),
            y=np.zeros((0,), dtype=np.float32),
            meta=[],
            extra_labels={},
            sampling={
                "limit": int(kwargs.get("limit") or 0),
                "max_samples": int(kwargs.get("max_samples") or 0),
                "sample_mode": str(kwargs.get("sample_mode") or ""),
            },
        )


def test_liquid_trainer_passes_sampling_controls_and_returns_manifest_context():
    fake = _FakePipeline()
    trainer = LiquidModelTrainer(
        pipeline=fake,  # type: ignore[arg-type]
        symbols=["BTC"],
        train_start="2025-01-01T00:00:00Z",
        train_end="2025-12-31T00:00:00Z",
        train_limit=4000,
        train_max_samples=512,
        train_sample_mode="tail",
    )
    out = trainer.train_symbol("BTC")
    assert out["status"] == "no_data"
    assert int(fake.kwargs["limit"]) == 4000
    assert int(fake.kwargs["max_samples"]) == 512
    assert str(fake.kwargs["sample_mode"]) == "tail"
    assert str(fake.kwargs["start"]).startswith("2025-01-01")
    assert str(fake.kwargs["end"]).startswith("2025-12-31")
    strategy = out.get("sampling_strategy") if isinstance(out, dict) else {}
    assert isinstance(strategy, dict)
    assert int(strategy.get("limit") or 0) == 4000
    assert int(strategy.get("max_samples") or 0) == 512


def test_research_mode_guardrails_filter_missing_and_downsample():
    fake = _FakePipeline()
    trainer = LiquidModelTrainer(
        pipeline=fake,  # type: ignore[arg-type]
        symbols=["BTC"],
        train_data_mode="research",
        train_max_samples=3,
    )
    n = 8
    d = len(LIQUID_FEATURE_KEYS)
    X = np.zeros((n, d), dtype=np.float32)
    missing_idx = [i for i, key in enumerate(LIQUID_FEATURE_KEYS) if str(key).endswith("_missing_flag")]
    assert len(missing_idx) > 0
    # First row has all modal features missing and should be filtered out in research mode.
    X[0, missing_idx] = 1.0
    y = np.arange(n, dtype=np.float32)
    batch = SampleBatch(
        X=X,
        y=y,
        meta=[{"as_of_ts_iso": f"t{i}"} for i in range(n)],
        extra_labels={"fwd_ret_1h": np.arange(n, dtype=np.float32)},
        sampling={},
    )
    out_batch, guard = trainer._apply_research_mode_guardrails(batch)
    assert bool(guard.get("enabled")) is True
    assert int(guard.get("rows_before") or 0) == n
    assert int(out_batch.X.shape[0]) <= 3
    assert int(out_batch.X.shape[0]) == int(out_batch.y.shape[0])
