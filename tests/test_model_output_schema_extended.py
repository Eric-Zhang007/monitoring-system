from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.liquid_model import LiquidModel, LiquidModelConfig


def test_model_output_schema_contains_distribution_and_router_fields():
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
        num_symbols=50,
        symbol_emb_dim=16,
        regime_dim=16,
        sparse_topk=2,
    )
    model = LiquidModel(cfg)
    xv = torch.randn(2, 16, 8)
    xm = torch.zeros(2, 16, 8)
    sid = torch.tensor([1, 7], dtype=torch.long)
    regime = torch.randn(2, 16)
    regime_mask = torch.zeros(2, 16)
    out = model(xv, xm, symbol_id=sid, regime_features=regime, regime_mask=regime_mask)
    assert tuple(out.mu.shape) == (2, 4)
    assert tuple(out.log_sigma.shape) == (2, 4)
    assert out.q is not None and tuple(out.q.shape) == (2, 4, 3)
    assert out.direction_logit is not None and tuple(out.direction_logit.shape) == (2, 4)
    assert out.expert_weights is not None and tuple(out.expert_weights.shape) == (2, 4)
    assert out.regime_probs is not None and tuple(out.regime_probs.shape) == (2, 3)
    dp = out.direction_prob()
    assert dp is not None and tuple(dp.shape) == (2, 4)
