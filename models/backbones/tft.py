from __future__ import annotations

import torch
from torch import nn

from models.backbones.base import BackboneBase


class VariableSelectionNetwork(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.selector(x), dim=-1)
        return x * w, w


class TFTBackbone(BackboneBase):
    def __init__(
        self,
        *,
        feature_dim: int,
        lookback: int,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.lookback = int(lookback)
        self.d_model = int(d_model)

        self.vsn = VariableSelectionNetwork(self.feature_dim)
        self.in_proj = nn.Linear(self.feature_dim, self.d_model)
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=max(1, int(n_heads)),
            dropout=float(dropout),
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(self.d_model)

    @property
    def out_dim(self) -> int:
        return int(self.d_model)

    def forward(self, x_values: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        if x_values.dim() != 3 or x_mask.dim() != 3:
            raise RuntimeError("tft_input_rank_invalid")
        if x_values.shape != x_mask.shape:
            raise RuntimeError("tft_values_mask_shape_mismatch")
        bsz, seq_len, feat_dim = x_values.shape
        if seq_len != self.lookback:
            raise RuntimeError(f"tft_lookback_mismatch:{seq_len}:{self.lookback}")
        if feat_dim != self.feature_dim:
            raise RuntimeError(f"tft_feature_dim_mismatch:{feat_dim}:{self.feature_dim}")

        xv = x_values.float()
        xm = x_mask.float().clamp(0.0, 1.0)
        obs = 1.0 - xm
        denom = obs.sum(dim=1, keepdim=True).clamp(min=1.0)
        f_mean = (xv * obs).sum(dim=1, keepdim=True) / denom
        xv_filled = xv * obs + (1.0 - obs) * f_mean

        x_sel, _weights = self.vsn(xv_filled)
        h_in = self.in_proj(x_sel)
        h_lstm, _ = self.lstm(h_in)

        time_missing = (xm.mean(dim=-1) >= 0.999)
        h_attn, _ = self.attn(h_lstm, h_lstm, h_lstm, key_padding_mask=time_missing)

        g = self.gate(torch.cat([h_lstm, h_attn], dim=-1))
        h = h_lstm + g * h_attn
        return self.norm(h)
