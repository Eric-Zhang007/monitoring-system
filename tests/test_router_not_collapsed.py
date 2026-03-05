from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.liquid_model import LiquidModel, LiquidModelConfig
from training.losses.trading_losses import compose_liquid_loss


def test_router_not_collapsed_after_minimal_training():
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
        num_symbols=16,
        symbol_emb_dim=16,
        regime_dim=16,
        sparse_topk=2,
    )
    model = LiquidModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(6):
        b = 24
        xv = torch.randn(b, 16, 8)
        xm = torch.zeros(b, 16, 8)
        sid = torch.randint(0, 16, (b,))
        regime = torch.randn(b, 16) * (1.0 + 0.1 * step)
        regime_mask = torch.zeros(b, 16)
        y = torch.randn(b, 4) * 0.01
        cost = torch.full((b, 4), 8.0)
        out = model(xv, xm, symbol_id=sid, regime_features=regime, regime_mask=regime_mask)
        losses = compose_liquid_loss(
            mu=out.mu,
            log_sigma=out.log_sigma,
            q=out.q,
            direction_logit=out.direction_logit,
            gate=out.aux.get("gate"),
            df=out.df,
            expert_weights=out.expert_weights,
            y=y,
            cost_bps=cost,
            quantiles=(0.1, 0.5, 0.9),
            w_nll=1.0,
            w_quantile=0.3,
            w_direction=0.2,
            w_gate=0.05,
            w_calibration=0.05,
            w_load_balance=0.05,
            w_router_entropy=0.02,
        )
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        opt.step()
    with torch.no_grad():
        out = model(
            torch.randn(64, 16, 8),
            torch.zeros(64, 16, 8),
            symbol_id=torch.randint(0, 16, (64,)),
            regime_features=torch.randn(64, 16),
            regime_mask=torch.zeros(64, 16),
        )
    assert out.expert_weights is not None
    usage = out.expert_weights.mean(dim=0)
    assert float(torch.max(usage)) < 0.95
