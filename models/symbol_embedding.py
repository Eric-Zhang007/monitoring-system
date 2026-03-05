from __future__ import annotations

import torch
from torch import nn


class SymbolEmbedding(nn.Module):
    def __init__(self, *, num_symbols: int, emb_dim: int):
        super().__init__()
        self.num_symbols = int(max(1, num_symbols))
        self.emb_dim = int(max(4, emb_dim))
        self.table = nn.Embedding(self.num_symbols, self.emb_dim)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(self, symbol_id: torch.Tensor) -> torch.Tensor:
        if symbol_id.dim() == 0:
            symbol_id = symbol_id.unsqueeze(0)
        if symbol_id.dim() != 1:
            raise RuntimeError(f"symbol_id_rank_invalid:{tuple(symbol_id.shape)}")
        sid = symbol_id.long().clamp(min=0, max=self.num_symbols - 1)
        return self.table(sid)
