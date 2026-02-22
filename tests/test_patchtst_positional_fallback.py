from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")

from models.backbones.patchtst import PatchTSTBackbone


def test_patchtst_long_lookback_uses_positional_fallback() -> None:
    lookback = 4096
    patch_len = 2
    n_patch = lookback // patch_len
    assert n_patch > 1024

    m = PatchTSTBackbone(
        feature_dim=4,
        lookback=lookback,
        d_model=16,
        n_layers=1,
        n_heads=4,
        dropout=0.1,
        patch_len=patch_len,
    )
    xv = torch.randn(2, lookback, 4, dtype=torch.float32)
    xm = torch.zeros(2, lookback, 4, dtype=torch.float32)
    h = m(xv, xm)
    assert tuple(h.shape) == (2, lookback, 16)
