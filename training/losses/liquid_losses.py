from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sigma = torch.exp(log_sigma).clamp(min=1e-6)
    z = (y - mu) / sigma
    return 0.5 * (z ** 2 + 2.0 * log_sigma)


def quantile_loss(q: torch.Tensor, y: torch.Tensor, taus: Sequence[float]) -> torch.Tensor:
    if q.dim() != 3:
        raise RuntimeError("quantile_loss_requires_rank3")
    y_exp = y.unsqueeze(-1)
    err = y_exp - q
    tau = torch.tensor(list(taus), device=q.device, dtype=q.dtype).view(1, 1, -1)
    return torch.maximum(tau * err, (tau - 1.0) * err)


def direction_bce(direction_logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_sign = (y >= 0.0).float()
    return F.binary_cross_entropy_with_logits(direction_logit, y_sign, reduction="none")


def gate_regularizer(gate: torch.Tensor, target_open_rate: float = 0.25) -> torch.Tensor:
    g_mean = gate.mean(dim=0)
    target = torch.full_like(g_mean, float(target_open_rate))
    return F.mse_loss(g_mean, target, reduction="mean")


def cost_weighting(cost_bps: torch.Tensor, strength: float = 0.08) -> torch.Tensor:
    return 1.0 / (1.0 + torch.clamp(cost_bps, min=0.0) * float(strength))


def compose_liquid_loss(
    *,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    q: torch.Tensor | None,
    direction_logit: torch.Tensor | None,
    gate: torch.Tensor | None,
    y: torch.Tensor,
    cost_bps: torch.Tensor,
    quantiles: Sequence[float],
    w_nll: float,
    w_quantile: float,
    w_direction: float,
    w_gate: float,
) -> Dict[str, torch.Tensor]:
    weights = cost_weighting(cost_bps).detach()

    nll = gaussian_nll(mu, log_sigma, y)
    nll_loss = (nll * weights).mean()

    q_loss = torch.tensor(0.0, device=y.device)
    if (q is not None) and (w_quantile > 0.0):
        qv = quantile_loss(q, y, quantiles)
        q_loss = qv.mean()

    d_loss = torch.tensor(0.0, device=y.device)
    if (direction_logit is not None) and (w_direction > 0.0):
        dv = direction_bce(direction_logit, y)
        d_loss = (dv * weights).mean()

    g_loss = torch.tensor(0.0, device=y.device)
    if (gate is not None) and (w_gate > 0.0):
        g_loss = gate_regularizer(gate)

    total = w_nll * nll_loss + w_quantile * q_loss + w_direction * d_loss + w_gate * g_loss
    return {
        "total": total,
        "gaussian_nll": nll_loss,
        "quantile": q_loss,
        "direction": d_loss,
        "gate": g_loss,
    }
