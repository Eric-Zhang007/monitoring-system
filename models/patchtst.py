from __future__ import annotations

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False


if HAS_TORCH:
    class PatchTSTBackbone(nn.Module):
        def __init__(self, feature_dim: int, lookback: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.lookback = int(lookback)
            self.input_dim = int(feature_dim) * 2  # values + mask
            self.proj = nn.Linear(self.input_dim, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=max(1, int(n_heads)),
                dim_feedforward=d_model * 4,
                dropout=float(dropout),
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(n_layers)))
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x_values: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
            x = torch.cat([x_values, x_mask], dim=-1)
            h = self.proj(x)
            z = self.encoder(h)
            return self.norm(z[:, -1, :])
else:  # pragma: no cover
    class PatchTSTBackbone:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("torch_required_for_patchtst")
