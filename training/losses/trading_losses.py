from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sigma = torch.exp(log_sigma).clamp(min=1e-6)
    z = (y - mu) / sigma
    return 0.5 * (z ** 2 + 2.0 * log_sigma)


def student_t_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor, df: torch.Tensor | None = None) -> torch.Tensor:
    sigma = torch.exp(log_sigma).clamp(min=1e-6)
    nu = torch.clamp(df if df is not None else torch.full_like(mu, 6.0), min=2.1, max=80.0)
    z2 = ((y - mu) / sigma) ** 2
    c = torch.lgamma((nu + 1.0) * 0.5) - torch.lgamma(nu * 0.5) - 0.5 * (torch.log(nu) + torch.log(torch.tensor(torch.pi, device=mu.device, dtype=mu.dtype)))
    ll = c - torch.log(sigma) - ((nu + 1.0) * 0.5) * torch.log1p(z2 / nu)
    return -ll


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


def calibration_regularizer(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # A simple sharpness-vs-error regularizer to keep predicted uncertainty calibrated.
    abs_err = (y - mu).abs()
    target = abs_err.detach().clamp(min=1e-6)
    return F.smooth_l1_loss(sigma, target, reduction="mean")


def load_balance_regularizer(expert_weights: torch.Tensor) -> torch.Tensor:
    if expert_weights.dim() != 2:
        raise RuntimeError("expert_weights_requires_rank2")
    usage = expert_weights.mean(dim=0)
    target = torch.full_like(usage, 1.0 / max(1, usage.numel()))
    return F.mse_loss(usage, target, reduction="mean")


def router_entropy_regularizer(expert_weights: torch.Tensor, min_entropy: float = 0.8) -> torch.Tensor:
    if expert_weights.dim() != 2:
        raise RuntimeError("expert_weights_requires_rank2")
    k = max(2, expert_weights.shape[-1])
    ent = (-expert_weights * torch.log(expert_weights.clamp(min=1e-8))).sum(dim=-1)
    ent_norm = ent / float(torch.log(torch.tensor(float(k), device=expert_weights.device)))
    shortfall = torch.relu(torch.tensor(float(min_entropy), device=expert_weights.device) - ent_norm)
    return shortfall.mean()


def horizon_smoothness_regularizer(mu: torch.Tensor) -> torch.Tensor:
    if mu.dim() != 2 or mu.shape[1] < 3:
        return torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
    second_diff = mu[:, 2:] - 2.0 * mu[:, 1:-1] + mu[:, :-2]
    return torch.mean(second_diff**2)


def vol_monotonic_regularizer(log_sigma: torch.Tensor) -> torch.Tensor:
    if log_sigma.dim() != 2 or log_sigma.shape[1] < 2:
        return torch.tensor(0.0, device=log_sigma.device, dtype=log_sigma.dtype)
    sigma = torch.exp(log_sigma).clamp(min=1e-6)
    shortfall = torch.relu(sigma[:, :-1] - sigma[:, 1:])
    return torch.mean(shortfall**2)


def compose_liquid_loss(
    *,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    q: torch.Tensor | None,
    direction_logit: torch.Tensor | None,
    gate: torch.Tensor | None,
    df: torch.Tensor | None = None,
    expert_weights: torch.Tensor | None = None,
    y: torch.Tensor,
    cost_bps: torch.Tensor,
    quantiles: Sequence[float],
    w_nll: float,
    w_quantile: float,
    w_direction: float,
    w_gate: float,
    w_calibration: float = 0.05,
    w_load_balance: float = 0.02,
    w_router_entropy: float = 0.01,
    w_horizon_smoothness: float = 0.02,
    w_vol_monotonic: float = 0.02,
) -> Dict[str, torch.Tensor]:
    weights = cost_weighting(cost_bps).detach()

    nll = student_t_nll(mu, log_sigma, y, df=df)
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

    sigma = torch.exp(log_sigma).clamp(min=1e-6)
    cal_loss = torch.tensor(0.0, device=y.device)
    if w_calibration > 0.0:
        cal_loss = calibration_regularizer(mu, sigma, y)

    lb_loss = torch.tensor(0.0, device=y.device)
    ent_loss = torch.tensor(0.0, device=y.device)
    if expert_weights is not None:
        if w_load_balance > 0.0:
            lb_loss = load_balance_regularizer(expert_weights)
        if w_router_entropy > 0.0:
            ent_loss = router_entropy_regularizer(expert_weights)
    hs_loss = torch.tensor(0.0, device=y.device)
    vm_loss = torch.tensor(0.0, device=y.device)
    if w_horizon_smoothness > 0.0:
        hs_loss = horizon_smoothness_regularizer(mu)
    if w_vol_monotonic > 0.0:
        vm_loss = vol_monotonic_regularizer(log_sigma)

    total = (
        w_nll * nll_loss
        + w_quantile * q_loss
        + w_direction * d_loss
        + w_gate * g_loss
        + w_calibration * cal_loss
        + w_load_balance * lb_loss
        + w_router_entropy * ent_loss
        + w_horizon_smoothness * hs_loss
        + w_vol_monotonic * vm_loss
    )
    return {
        "total": total,
        "gaussian_nll": nll_loss,
        "student_t_nll": nll_loss,
        "quantile": q_loss,
        "direction": d_loss,
        "gate": g_loss,
        "calibration": cal_loss,
        "load_balance": lb_loss,
        "router_entropy": ent_loss,
        "horizon_smoothness": hs_loss,
        "vol_monotonic": vm_loss,
    }
