from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")


from models.heads.dist_head import MultiHorizonDistHead


def test_gate_behavior_respects_text_coverage():
    head = MultiHorizonDistHead(
        hidden_dim=16,
        horizons=("1h", "4h", "1d", "7d"),
        quantiles=(0.1, 0.5, 0.9),
        text_indices=(0, 1, 2, 3),
        quality_indices=(4, 5),
    )
    h_seq = torch.randn(2, 8, 16)

    xv = torch.randn(2, 8, 6)
    xm_missing = torch.zeros(2, 8, 6)
    xm_missing[:, :, 0:4] = 1.0  # text fully missing
    out_missing = head(h_seq, xv, xm_missing)
    gate_missing = out_missing.aux["gate"]

    xm_present = torch.zeros(2, 8, 6)
    out_present = head(h_seq, xv, xm_present)
    gate_present = out_present.aux["gate"]

    assert torch.mean(gate_missing) < torch.mean(gate_present)
