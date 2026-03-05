from __future__ import annotations

import torch

from training.losses.trading_losses import horizon_smoothness_regularizer, vol_monotonic_regularizer


def test_horizon_smoothness_regularizer_positive_on_jagged_curve():
    mu = torch.tensor([[0.01, -0.02, 0.03, -0.01]], dtype=torch.float32)
    val = horizon_smoothness_regularizer(mu)
    assert float(val.item()) > 0.0


def test_vol_monotonic_regularizer_positive_on_descending_sigma():
    sigma = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
    log_sigma = torch.log(sigma)
    val = vol_monotonic_regularizer(log_sigma)
    assert float(val.item()) > 0.0

