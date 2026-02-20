from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from summarize_phase_d_results import build_summary  # noqa: E402


def test_build_summary_collects_ready_backbones_and_ablations():
    out = build_summary(
        candidate_report={"model_name": "multimodal_ridge", "fusion_mode": "residual_gate"},
        backbone_report={
            "torch_available": True,
            "results": [
                {"backbone": "ridge", "ready": True, "walk_forward": {"status": "ok", "folds": 4, "basic": {"mse": 0.1, "mae": 0.2, "hit_rate": 0.55}}},
                {"backbone": "tft", "ready": False, "walk_forward": {"status": "blocked", "reason": "torch_missing", "folds": 0}},
            ],
        },
        eval_report={
            "primary_ablation": "full",
            "ablation_results": [
                {"ablation": "full", "status": "ok", "rows": 1000, "folds": 4, "mse": 0.1, "mae": 0.2, "hit_rate": 0.55},
                {"ablation": "no_text", "status": "ok", "rows": 1000, "folds": 4, "mse": 0.12, "mae": 0.21, "hit_rate": 0.53},
            ],
        },
    )
    assert out["candidate"]["fusion_mode"] == "residual_gate"
    assert out["backbone"]["ready_backbones"] == ["ridge"]
    assert out["ablation"]["primary_ablation"] == "full"
    assert len(out["ablation"]["rows"]) == 2
