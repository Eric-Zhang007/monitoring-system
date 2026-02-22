from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class QualityEncoder(nn.Module):
    def __init__(self, quality_indices: Sequence[int], d_q: int = 32):
        super().__init__()
        self.quality_indices = [int(x) for x in quality_indices]
        self.input_dim = len(self.quality_indices)
        self.d_q = int(d_q)
        if self.input_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, max(16, self.d_q)),
                nn.GELU(),
                nn.Linear(max(16, self.d_q), self.d_q),
            )
        else:
            self.net = None

    def forward(self, x_values: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _feat_dim = x_values.shape
        if self.input_dim <= 0:
            return torch.zeros((bsz, self.d_q), device=x_values.device, dtype=x_values.dtype)

        xv = x_values[:, :, self.quality_indices]
        xm = x_mask[:, :, self.quality_indices].float().clamp(0.0, 1.0)
        obs = 1.0 - xm

        decay = torch.linspace(0.25, 1.0, seq_len, device=x_values.device, dtype=x_values.dtype).view(1, seq_len, 1)
        w = obs * decay
        denom = w.sum(dim=1).clamp(min=1.0)
        pooled = (xv * w).sum(dim=1) / denom
        return self.net(pooled)
