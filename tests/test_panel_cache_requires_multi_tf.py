from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.datasets.liquid_panel_cache_dataset import LiquidPanelCacheDataset


def test_panel_cache_fail_fast_when_multi_tf_manifest_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("LIQUID_REQUIRE_MULTI_TF_CONTEXT", "1")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    t = 64
    f = 8
    np.savez_compressed(
        cache_dir / "AAA.npz",
        values=np.zeros((t, f), dtype=np.float32),
        mask=np.zeros((t, f), dtype=np.uint8),
        close=np.linspace(100.0, 120.0, t, dtype=np.float64),
        end_ts=np.arange(1_700_000_000, 1_700_000_000 + t * 300, 300, dtype=np.int64),
        regime_features=np.zeros((t, 16), dtype=np.float32),
        regime_mask=np.zeros((t, 16), dtype=np.uint8),
    )
    np.savez_compressed(
        cache_dir / "index.npz",
        symbol_id=np.asarray([0], dtype=np.int32),
        t_idx=np.asarray([40], dtype=np.int32),
        end_ts=np.asarray([1_700_012_000], dtype=np.int64),
    )
    manifest = {
        "bar_size": "5m",
        "symbols": ["AAA"],
        "symbol_to_id": {"AAA": 0},
        "regime_feature_names": [f"r{i}" for i in range(16)],
        "index_file": "index.npz",
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="training_cache_multi_tf_feature_names_missing"):
        LiquidPanelCacheDataset(cache_dir=cache_dir, lookback=32)
