from __future__ import annotations

import torch
from torch import nn

from models.backbones.base import BackboneBase


class ITransformerBackbone(BackboneBase):
    def __init__(
        self,
        *,
        feature_dim: int,
        lookback: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.lookback = int(lookback)
        self.d_model = int(d_model)

        self.time_proj = nn.Linear(self.lookback, self.d_model)
        self.mask_embed = nn.Linear(1, self.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=max(1, int(n_heads)),
            dim_feedforward=self.d_model * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(n_layers)))
        self.token_weight = nn.Linear(self.d_model, 1)
        self.time_context = nn.Linear(self.feature_dim, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

    @property
    def out_dim(self) -> int:
        return int(self.d_model)

    def forward(self, x_values: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        if x_values.dim() != 3 or x_mask.dim() != 3:
            raise RuntimeError("itransformer_input_rank_invalid")
        if x_values.shape != x_mask.shape:
            raise RuntimeError("itransformer_values_mask_shape_mismatch")
        bsz, seq_len, feat_dim = x_values.shape
        if seq_len != self.lookback:
            raise RuntimeError(f"itransformer_lookback_mismatch:{seq_len}:{self.lookback}")
        if feat_dim != self.feature_dim:
            raise RuntimeError(f"itransformer_feature_dim_mismatch:{feat_dim}:{self.feature_dim}")

        xv = x_values.float()
        xm = x_mask.float().clamp(0.0, 1.0)
        obs = (1.0 - xm)

        denom = obs.sum(dim=1, keepdim=True).clamp(min=1.0)
        f_mean = (xv * obs).sum(dim=1, keepdim=True) / denom
        xv_filled = xv * obs + (1.0 - obs) * f_mean

        var_tokens = xv_filled.transpose(1, 2)
        var_mask = xm.transpose(1, 2).mean(dim=-1, keepdim=True)

        tok = self.time_proj(var_tokens) + self.mask_embed(1.0 - var_mask)
        key_padding = (var_mask.squeeze(-1) >= 0.999)
        h_var = self.encoder(tok, src_key_padding_mask=key_padding)

        alpha = torch.softmax(self.token_weight(h_var), dim=1)
        global_ctx = (h_var * alpha).sum(dim=1)

        h_seq = self.time_context(xv_filled) + global_ctx.unsqueeze(1)
        return self.norm(h_seq)
