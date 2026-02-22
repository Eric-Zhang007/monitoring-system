from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")


from models.heads.dist_head import MultiHorizonDistHead


def test_dist_head_shapes_stable():
    head = MultiHorizonDistHead(
        hidden_dim=32,
        horizons=("1h", "4h", "1d", "7d"),
        quantiles=(0.1, 0.5, 0.9),
        text_indices=(0, 1, 2),
        quality_indices=(3, 4),
    )
    h_seq = torch.randn(3, 12, 32)
    xv = torch.randn(3, 12, 6)
    xm = torch.zeros(3, 12, 6)

    out = head(h_seq, xv, xm)
    assert tuple(out.mu.shape) == (3, 4)
    assert tuple(out.log_sigma.shape) == (3, 4)
    assert out.q is not None and tuple(out.q.shape) == (3, 4, 3)
    assert out.direction_logit is not None and tuple(out.direction_logit.shape) == (3, 4)
