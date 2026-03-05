from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from torch import nn

from models.backbones.registry import build_backbone
from models.heads.dist_head import MultiHorizonDistHead
from models.outputs import MultiHorizonDistOutput
from models.symbol_embedding import SymbolEmbedding


DEFAULT_HORIZONS = ("1h", "4h", "1d", "7d")
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class LiquidModelConfig:
    backbone_name: str
    lookback: int
    feature_dim: int
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    patch_len: int
    horizons: Sequence[str]
    quantiles: Sequence[float]
    text_indices: Sequence[int]
    quality_indices: Sequence[int]
    num_symbols: int = 1
    symbol_emb_dim: int = 16
    regime_dim: int = 16
    sparse_topk: int = 2


class LiquidModel(nn.Module):
    def __init__(self, cfg: LiquidModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(
            cfg.backbone_name,
            feature_dim=int(cfg.feature_dim),
            lookback=int(cfg.lookback),
            d_model=int(cfg.d_model),
            n_layers=int(cfg.n_layers),
            n_heads=int(cfg.n_heads),
            dropout=float(cfg.dropout),
            patch_len=int(cfg.patch_len),
        )
        self.head = MultiHorizonDistHead(
            hidden_dim=int(self.backbone.out_dim),
            horizons=cfg.horizons,
            quantiles=cfg.quantiles,
            text_indices=cfg.text_indices,
            quality_indices=cfg.quality_indices,
            symbol_dim=int(cfg.symbol_emb_dim),
            regime_dim=int(cfg.regime_dim),
            sparse_topk=int(cfg.sparse_topk),
        )
        self.symbol_embedding = SymbolEmbedding(num_symbols=int(cfg.num_symbols), emb_dim=int(cfg.symbol_emb_dim))

    def forward(
        self,
        x_values: torch.Tensor,
        x_mask: torch.Tensor,
        *,
        symbol_id: torch.Tensor | None = None,
        regime_features: torch.Tensor | None = None,
        regime_mask: torch.Tensor | None = None,
        router_hint: torch.Tensor | None = None,
    ) -> MultiHorizonDistOutput:
        h_seq = self.backbone(x_values, x_mask)
        if symbol_id is None:
            symbol_id = torch.zeros((x_values.shape[0],), dtype=torch.long, device=x_values.device)
        symbol_ctx = self.symbol_embedding(symbol_id)
        return self.head(
            h_seq,
            x_values,
            x_mask,
            static_ctx=symbol_ctx,
            regime_features=regime_features,
            regime_mask=regime_mask,
            router_hint=router_hint,
        )

    def export_meta(self) -> Dict[str, object]:
        return {
            "backbone_name": str(self.cfg.backbone_name),
            "lookback": int(self.cfg.lookback),
            "feature_dim": int(self.cfg.feature_dim),
            "d_model": int(self.cfg.d_model),
            "n_layers": int(self.cfg.n_layers),
            "n_heads": int(self.cfg.n_heads),
            "dropout": float(self.cfg.dropout),
            "patch_len": int(self.cfg.patch_len),
            "horizons": [str(h) for h in self.cfg.horizons],
            "quantiles": [float(q) for q in self.cfg.quantiles],
            "text_indices": [int(x) for x in self.cfg.text_indices],
            "quality_indices": [int(x) for x in self.cfg.quality_indices],
            "num_symbols": int(self.cfg.num_symbols),
            "symbol_emb_dim": int(self.cfg.symbol_emb_dim),
            "regime_dim": int(self.cfg.regime_dim),
            "sparse_topk": int(self.cfg.sparse_topk),
        }


def build_liquid_model_from_checkpoint(ckpt: Dict[str, object]) -> LiquidModel:
    cfg = LiquidModelConfig(
        backbone_name=str(ckpt.get("backbone_name") or "patchtst"),
        lookback=int(ckpt.get("lookback") or 0),
        feature_dim=int(ckpt.get("feature_dim") or 0),
        d_model=int(ckpt.get("d_model") or 128),
        n_layers=int(ckpt.get("n_layers") or 2),
        n_heads=int(ckpt.get("n_heads") or 4),
        dropout=float(ckpt.get("dropout") or 0.1),
        patch_len=int(ckpt.get("patch_len") or 16),
        horizons=list(ckpt.get("horizons") or list(DEFAULT_HORIZONS)),
        quantiles=list(ckpt.get("quantiles") or list(DEFAULT_QUANTILES)),
        text_indices=list(ckpt.get("text_indices") or []),
        quality_indices=list(ckpt.get("quality_indices") or []),
        num_symbols=int(ckpt.get("num_symbols") or 1),
        symbol_emb_dim=int(ckpt.get("symbol_emb_dim") or 16),
        regime_dim=int(ckpt.get("regime_dim") or 16),
        sparse_topk=int(ckpt.get("sparse_topk") or 2),
    )
    model = LiquidModel(cfg)
    state = ckpt.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("weights_state_dict_missing")
    model.load_state_dict(state)
    return model
