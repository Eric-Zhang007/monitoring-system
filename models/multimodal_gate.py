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
    class ResidualGateHead(nn.Module):
        def __init__(self, hidden_dim: int, text_dim: int, quality_dim: int, out_dim: int = 4):
            super().__init__()
            self.base = nn.Linear(hidden_dim, out_dim)
            self.delta = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )
            self.gate = nn.Sequential(
                nn.Linear(quality_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        def forward(self, h_num: torch.Tensor, text_vec: torch.Tensor, quality_vec: torch.Tensor):
            base = self.base(h_num)
            delta = self.delta(text_vec)
            gate = self.gate(quality_vec)
            out = base + gate * delta
            return out, gate
else:  # pragma: no cover
    class ResidualGateHead:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("torch_required_for_residual_gate")
