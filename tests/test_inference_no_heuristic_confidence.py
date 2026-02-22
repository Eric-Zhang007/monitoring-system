from __future__ import annotations

from pathlib import Path


def test_inference_no_abs_pred_heuristic_confidence():
    text = Path("backend/liquid_model_service.py").read_text(encoding="utf-8")
    assert "0.5 + abs" not in text
    assert "vol_map = {k: float(max(1e-6, abs(v)))" not in text
