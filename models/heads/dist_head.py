from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import nn

from models.outputs import MultiHorizonDistOutput
from models.quality_encoder import QualityEncoder
from models.text_tower import TextTower


class MultiHorizonDistHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        horizons: Sequence[str],
        quantiles: Sequence[float],
        text_indices: Sequence[int],
        quality_indices: Sequence[int],
        d_text: int = 64,
        d_q: int = 32,
        gate_bias: float = -1.5,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.horizons = [str(h) for h in horizons]
        self.quantiles = [float(q) for q in quantiles]
        self.h_count = len(self.horizons)
        self.q_count = len(self.quantiles)

        self.base_mu = nn.Linear(self.hidden_dim, self.h_count)
        self.log_sigma = nn.Linear(self.hidden_dim, self.h_count)
        self.direction = nn.Linear(self.hidden_dim, self.h_count)

        self.text_tower = TextTower(text_indices=text_indices, d_text=d_text)
        self.quality_encoder = QualityEncoder(quality_indices=quality_indices, d_q=d_q)

        self.delta_mu = nn.Sequential(
            nn.Linear(d_text, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.h_count),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_q, max(16, d_q)),
            nn.GELU(),
            nn.Linear(max(16, d_q), self.h_count),
        )
        self.gate_bias = nn.Parameter(torch.full((self.h_count,), float(gate_bias)))

        if self.q_count > 0:
            self.quantile_delta = nn.Linear(self.hidden_dim, self.h_count * self.q_count)
        else:
            self.quantile_delta = None

    def forward(
        self,
        h_seq: torch.Tensor,
        x_values: torch.Tensor,
        x_mask: torch.Tensor,
        group_slices: Optional[Dict[str, Iterable[int]]] = None,
    ) -> MultiHorizonDistOutput:
        _ = group_slices
        h_last = h_seq[:, -1, :]

        mu_base = self.base_mu(h_last)
        log_sigma = self.log_sigma(h_last).clamp(min=-7.0, max=2.0)
        direction_logit = self.direction(h_last)

        t_pool = self.text_tower(x_values, x_mask)
        q_vec = self.quality_encoder(x_values, x_mask)
        delta_mu = self.delta_mu(t_pool)

        raw_gate = torch.sigmoid(self.gate(q_vec) + self.gate_bias)

        text_missing = torch.ones((x_values.shape[0], 1), device=x_values.device, dtype=x_values.dtype)
        if self.text_tower.input_dim > 0:
            xm = x_mask[:, :, self.text_tower.text_indices].float().clamp(0.0, 1.0)
            coverage = 1.0 - xm.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
            text_missing = (coverage > 0).float()
        gate = raw_gate * text_missing

        mu = mu_base + gate * delta_mu

        q_out = None
        if self.quantile_delta is not None:
            sigma = torch.exp(log_sigma)
            delta = self.quantile_delta(h_last).view(-1, self.h_count, self.q_count)
            q_out = mu.unsqueeze(-1) + sigma.unsqueeze(-1) * delta
            q_out, _ = torch.sort(q_out, dim=-1)

        aux = {
            "gate": gate,
            "gate_raw": raw_gate,
            "text_available": text_missing,
            "text_pool": t_pool,
            "quality_vec": q_vec,
        }
        return MultiHorizonDistOutput(
            mu=mu,
            log_sigma=log_sigma,
            q=q_out,
            direction_logit=direction_logit,
            aux=aux,
        )
