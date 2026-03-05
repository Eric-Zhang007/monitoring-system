from __future__ import annotations

import numpy as np
import torch

from training.calibration.calibrate import build_calibration_bundle
from training.losses.trading_losses import compose_liquid_loss


def test_training_loss_outputs_and_calibration_bundle():
    torch.manual_seed(7)
    b, h, qn = 8, 4, 3
    mu = torch.randn(b, h)
    log_sigma = torch.randn(b, h) * 0.1
    q = torch.randn(b, h, qn)
    direction_logit = torch.randn(b, h)
    gate = torch.sigmoid(torch.randn(b, h))
    y = torch.randn(b, h)
    cost_bps = torch.rand(b, h) * 12.0

    losses = compose_liquid_loss(
        mu=mu,
        log_sigma=log_sigma,
        q=q,
        direction_logit=direction_logit,
        gate=gate,
        y=y,
        cost_bps=cost_bps,
        quantiles=[0.1, 0.5, 0.9],
        w_nll=1.0,
        w_quantile=0.3,
        w_direction=0.2,
        w_gate=0.05,
    )
    for k in ("total", "gaussian_nll", "quantile", "direction", "gate", "horizon_smoothness", "vol_monotonic"):
        assert k in losses
        assert torch.isfinite(losses[k]).item()

    cal = build_calibration_bundle(
        direction_logit=direction_logit.detach().cpu().numpy(),
        y_net=y.detach().cpu().numpy(),
        mu=mu.detach().cpu().numpy(),
        sigma=np.exp(log_sigma.detach().cpu().numpy()),
    )
    assert float(cal.get("sigma_scale", 0.0)) > 0.0
    assert float(cal.get("direction_temperature", 0.0)) > 0.0
