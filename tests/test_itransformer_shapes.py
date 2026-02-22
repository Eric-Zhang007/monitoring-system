from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")


from models.backbones.itransformer import ITransformerBackbone


def test_itransformer_shapes():
    m = ITransformerBackbone(feature_dim=7, lookback=12, d_model=24, n_layers=1, n_heads=4, dropout=0.1)
    xv = torch.randn(4, 12, 7)
    xm = torch.zeros(4, 12, 7)
    h = m(xv, xm)
    assert tuple(h.shape) == (4, 12, 24)
