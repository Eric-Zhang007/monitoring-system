from __future__ import annotations

import pytest
pytest.importorskip("torch")

from models.backbones.registry import build_backbone


@pytest.mark.parametrize("name", ["patchtst", "itransformer", "tft"])
def test_backbone_registry_build_ok(name: str):
    b = build_backbone(
        name,
        feature_dim=8,
        lookback=16,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.1,
        patch_len=4,
    )
    assert b is not None


def test_backbone_registry_unknown_raises():
    with pytest.raises(ValueError):
        build_backbone("unknown", feature_dim=8, lookback=16, d_model=32, n_layers=1, n_heads=4, dropout=0.1, patch_len=4)
