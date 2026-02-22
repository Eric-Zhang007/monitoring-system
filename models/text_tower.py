from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class TextTower(nn.Module):
    def __init__(self, text_indices: Sequence[int], d_text: int = 64):
        super().__init__()
        self.text_indices = [int(x) for x in text_indices]
        self.input_dim = len(self.text_indices)
        self.d_text = int(d_text)
        if self.input_dim > 0:
            self.proj = nn.Linear(self.input_dim, self.d_text)
            self.gru = nn.GRU(self.d_text, self.d_text, num_layers=1, batch_first=True)
            self.attn = nn.Linear(self.d_text, 1)
        else:
            self.proj = None
            self.gru = None
            self.attn = None

    def forward(self, x_values: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        bsz, _seq_len, _feat_dim = x_values.shape
        if self.input_dim <= 0:
            return torch.zeros((bsz, self.d_text), device=x_values.device, dtype=x_values.dtype)

        xv = x_values[:, :, self.text_indices]
        xm = x_mask[:, :, self.text_indices].float().clamp(0.0, 1.0)
        obs = 1.0 - xm
        denom = obs.sum(dim=-1, keepdim=True).clamp(min=1.0)
        step_cov = obs.mean(dim=-1, keepdim=True)
        x_step = (xv * obs).sum(dim=-1, keepdim=True) / denom

        h = self.proj(xv * obs)
        h, _ = self.gru(h)
        score = self.attn(h) + torch.log(step_cov.clamp(min=1e-6))
        weight = torch.softmax(score, dim=1)
        pooled = (h * weight).sum(dim=1)

        # If the full text window is missing, return zeros to let gate fully close.
        text_cov = step_cov.mean(dim=1)
        return pooled * (text_cov > 0).float()
