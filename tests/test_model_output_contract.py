from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")


from models.liquid_model import LiquidModel, LiquidModelConfig
from models.outputs import MultiHorizonDistOutput


def test_model_output_contract():
    cfg = LiquidModelConfig(
        backbone_name="patchtst",
        lookback=16,
        feature_dim=8,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.1,
        patch_len=4,
        horizons=("1h", "4h", "1d", "7d"),
        quantiles=(0.1, 0.5, 0.9),
        text_indices=(0, 1),
        quality_indices=(2, 3),
    )
    model = LiquidModel(cfg)
    xv = torch.randn(2, 16, 8)
    xm = torch.zeros(2, 16, 8)

    out = model(xv, xm)
    assert isinstance(out, MultiHorizonDistOutput)
    assert tuple(out.mu.shape) == (2, 4)
    assert tuple(out.log_sigma.shape) == (2, 4)
    assert out.direction_logit is not None
    assert tuple(out.direction_logit.shape) == (2, 4)
    assert out.q is not None
    assert tuple(out.q.shape) == (2, 4, 3)
