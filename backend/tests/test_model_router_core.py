from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "inference"))

from model_router import ModelRouter  # noqa: E402


def test_align_features_strict_dim_match():
    x = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="feature_dim_mismatch"):
        ModelRouter._align_features(x, 4)
    exact = ModelRouter._align_features(x, 2)
    assert exact.shape[0] == 2
    assert float(exact[0]) == 1.0
    assert float(exact[1]) == 2.0


def test_normalize_features_uses_stats():
    x = np.array([10.0, 5.0, -1.0], dtype=np.float32)
    norm = {"x_mean": [8.0, 1.0, -1.0], "x_std": [2.0, 2.0, 0.5]}
    out = ModelRouter._normalize_features(x, norm)
    assert np.allclose(out, np.array([1.0, 2.0, 0.0], dtype=np.float32))


def test_schema_compatibility_includes_previous_v22():
    router = ModelRouter()
    assert "v2.2" in router.compatible_feature_schemas
