from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")


from models.backbones.tft import TFTBackbone


def test_tft_forward_and_backward():
    m = TFTBackbone(feature_dim=6, lookback=14, d_model=20, n_heads=4, dropout=0.1)
    xv = torch.randn(3, 14, 6, requires_grad=True)
    xm = torch.zeros(3, 14, 6)
    h = m(xv, xm)
    loss = h.mean()
    loss.backward()
    assert tuple(h.shape) == (3, 14, 20)
    assert xv.grad is not None
