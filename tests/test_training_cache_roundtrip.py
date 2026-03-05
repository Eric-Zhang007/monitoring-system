from __future__ import annotations

from pathlib import Path

import numpy as np

from training.cache.panel_cache import write_toy_cache
from training.datasets.liquid_panel_cache_dataset import LiquidPanelCacheDataset


def test_training_cache_roundtrip_shapes_and_no_leakage(tmp_path: Path):
    symbols = ["AAA", "BBB"]
    t = 200
    f = 12
    end_ts = np.arange(1_700_000_000, 1_700_000_000 + t * 300, 300, dtype=np.int64)
    values_map = {}
    mask_map = {}
    close_map = {}
    end_ts_map = {}
    for i, sym in enumerate(symbols):
        vals = np.zeros((t, f), dtype=np.float32)
        vals[:, 0] = np.linspace(0.0, 1.0 + i, t, dtype=np.float32)
        vals[:, 1] = np.sin(np.linspace(0.0, 8.0, t)).astype(np.float32)
        vals[:, 2] = 10.0 + i
        vals[:, 3] = 0.2
        vals[:, 4] = 1000.0
        vals[:, 5] = 0.1
        vals[:, 6] = 0.01
        vals[:, 7] = 0.0
        vals[:, 8] = 100.0
        vals[:, 9] = 0.3
        vals[:, 10] = 0.2
        vals[:, 11] = 1.0
        msk = np.zeros((t, f), dtype=np.uint8)
        msk[::17, 10] = 1
        close = 100.0 + np.arange(t, dtype=np.float64) * (0.1 + i * 0.02)
        values_map[sym] = vals
        mask_map[sym] = msk
        close_map[sym] = close
        end_ts_map[sym] = end_ts.copy()

    cache_dir = tmp_path / "cache"
    write_toy_cache(
        output_dir=cache_dir,
        symbols=symbols,
        values_map=values_map,
        mask_map=mask_map,
        close_map=close_map,
        end_ts_map=end_ts_map,
        lookback_len=32,
        horizons=("1h", "4h"),
        universe_snapshot_hash="toy_hash_001",
        bar_size="5m",
    )

    ds = LiquidPanelCacheDataset(cache_dir=cache_dir, lookback=32, horizons=("1h", "4h"))
    assert len(ds) >= 10
    expected_mtf_dim = len(ds.manifest.get("multi_tf_feature_names") or [])
    for i in range(10):
        sample = ds[i]
        assert tuple(sample["x_values"].shape) == (32, f)
        assert tuple(sample["x_mask"].shape) == (32, f)
        assert tuple(sample["y_raw"].shape) == (2,)
        assert tuple(sample["cost_bps"].shape) == (2,)
        assert tuple(sample["multi_timeframe_context"].shape) == (expected_mtf_dim,)
        assert tuple(sample["multi_timeframe_mask"].shape) == (expected_mtf_dim,)
        assert tuple(sample["regime_features"].shape) == (16 + expected_mtf_dim,)
        assert float(sample["x_values"][-1, 0].item()) >= float(sample["x_values"][0, 0].item())
        assert float(sample["y_raw"][0].item()) > 0.0
