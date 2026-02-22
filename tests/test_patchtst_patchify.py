from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")


from models.backbones.patchtst import PatchTSTBackbone


def test_patchtst_patchify_padding_and_shape():
    m = PatchTSTBackbone(feature_dim=5, lookback=10, d_model=16, n_layers=1, n_heads=4, dropout=0.1, patch_len=4)
    xv = torch.randn(2, 10, 5)
    xm = torch.zeros(2, 10, 5)
    pv, pm, pad = m._patchify(xv, xm)

    assert pad == 2
    assert tuple(pv.shape) == (2, 5, 3, 4)
    assert tuple(pm.shape) == (2, 5, 3, 4)
    # padded tail must be missing mask=1
    assert torch.all(pm[:, :, -1, -2:] == 1)
