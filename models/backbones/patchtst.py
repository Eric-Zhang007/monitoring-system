from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from models.backbones.base import BackboneBase


class RevIN(nn.Module):
    """Masked RevIN normalization (instance-level)."""

    def __init__(self, feature_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.ones(1, 1, int(feature_dim)))
        self.beta = nn.Parameter(torch.zeros(1, 1, int(feature_dim)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = (1.0 - mask).clamp(0.0, 1.0)
        denom = obs.sum(dim=1, keepdim=True).clamp(min=1.0)
        mu = (x * obs).sum(dim=1, keepdim=True) / denom
        var = (((x - mu) * obs) ** 2).sum(dim=1, keepdim=True) / denom
        std = torch.sqrt(var + self.eps)
        z = ((x - mu) / std) * self.gamma + self.beta
        z = z * obs
        return z, mu, std


class PatchTSTBackbone(BackboneBase):
    def __init__(
        self,
        *,
        feature_dim: int,
        lookback: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        patch_len: int = 16,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.lookback = int(lookback)
        self.patch_len = max(2, int(patch_len))
        self.d_model = int(d_model)
        self.revin = RevIN(feature_dim=self.feature_dim)

        self.patch_embed = nn.Linear(self.patch_len, self.d_model)
        self.mask_embed = nn.Linear(1, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, self.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=max(1, int(n_heads)),
            dim_feedforward=self.d_model * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(n_layers)))
        self.norm = nn.LayerNorm(self.d_model)

    @property
    def out_dim(self) -> int:
        return int(self.d_model)

    def _patchify(
        self,
        x_values: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        bsz, seq_len, feat_dim = x_values.shape
        if feat_dim != self.feature_dim:
            raise RuntimeError(f"patchtst_feature_dim_mismatch:{feat_dim}:{self.feature_dim}")
        pad_len = (self.patch_len - (seq_len % self.patch_len)) % self.patch_len
        if pad_len > 0:
            x_values = torch.cat(
                [x_values, torch.zeros((bsz, pad_len, feat_dim), device=x_values.device, dtype=x_values.dtype)],
                dim=1,
            )
            x_mask = torch.cat(
                [x_mask, torch.ones((bsz, pad_len, feat_dim), device=x_mask.device, dtype=x_mask.dtype)],
                dim=1,
            )
        n_patch = x_values.shape[1] // self.patch_len
        xv = x_values.transpose(1, 2).contiguous().view(bsz, feat_dim, n_patch, self.patch_len)
        xm = x_mask.transpose(1, 2).contiguous().view(bsz, feat_dim, n_patch, self.patch_len)
        return xv, xm, pad_len

    def forward(self, x_values: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        if x_values.dim() != 3 or x_mask.dim() != 3:
            raise RuntimeError("patchtst_input_rank_invalid")
        if x_values.shape != x_mask.shape:
            raise RuntimeError("patchtst_values_mask_shape_mismatch")

        x_values = x_values.float()
        x_mask = x_mask.float().clamp(0.0, 1.0)
        x_norm, _mu, _std = self.revin(x_values, x_mask)
        xv, xm, pad_len = self._patchify(x_norm, x_mask)

        bsz, feat_dim, n_patch, _ = xv.shape
        patch_obs = 1.0 - xm.mean(dim=-1, keepdim=True)
        patch_values = xv * (1.0 - xm)

        tokens = self.patch_embed(patch_values.view(bsz * feat_dim * n_patch, self.patch_len))
        tokens = tokens.view(bsz * feat_dim, n_patch, self.d_model)

        if n_patch > self.pos_embed.shape[1]:
            raise RuntimeError(f"patchtst_positional_capacity_exceeded:{n_patch}:{self.pos_embed.shape[1]}")
        pos = self.pos_embed[:, :n_patch, :]
        mask_tok = self.mask_embed(patch_obs.view(bsz * feat_dim, n_patch, 1))
        src_key_padding_mask = (patch_obs.view(bsz * feat_dim, n_patch) <= 1e-6)

        h = self.encoder(tokens + pos + mask_tok, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)

        h_patch = h.view(bsz, feat_dim, n_patch, self.d_model)
        h_time = (
            h_patch.unsqueeze(3)
            .expand(bsz, feat_dim, n_patch, self.patch_len, self.d_model)
            .reshape(bsz, feat_dim, n_patch * self.patch_len, self.d_model)
        )
        h_seq = h_time.mean(dim=1)
        if pad_len > 0:
            h_seq = h_seq[:, :-pad_len, :]
        return h_seq
