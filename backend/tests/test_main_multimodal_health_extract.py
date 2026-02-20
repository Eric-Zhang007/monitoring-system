from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DATABASE_URL", "postgresql://monitor:monitor@127.0.0.1:1/monitor?connect_timeout=1")
sys.path.append(str(ROOT / "backend"))

import main as main_mod  # noqa: E402


def test_extract_multimodal_health_prefers_runtime_coverage_and_candidate_gate():
    out = main_mod._extract_multimodal_health(
        latest_candidate={
            "fusion_mode": "residual_gate",
            "text_dropout_prob": 0.2,
            "gate_val_mean": 0.65,
            "delta_val_mean_abs": 0.03,
            "backbone_ready_list": ["ridge", "itransformer"],
            "gate": {"passed": True, "reasons": []},
            "ablation_summary": {
                "primary": "full",
                "delta_mse_no_text_vs_full": 0.04,
                "rows": {"full": 1000, "event_window": 200},
            },
        },
        phase_d_summary={},
        social_throughput={"text_coverage_ratio": 0.44},
    )
    assert out["fusion_mode"] == "residual_gate"
    assert abs(float(out["text_coverage_ratio"]) - 0.44) < 1e-12
    assert abs(float(out["gate_openness"]) - 0.65) < 1e-12
    assert int(out["ready_backbone_count"]) == 2
    assert out["candidate_gate_passed"] is True


def test_extract_multimodal_health_falls_back_to_ablation_and_summary_backbones():
    out = main_mod._extract_multimodal_health(
        latest_candidate={
            "gate_val_mean": 1.5,
            "delta_val_mean_abs": 0.0,
            "ablation_summary": {
                "rows": {"full": 240, "event_window": 60},
                "delta_mse_no_text_vs_full": 0.08,
            },
        },
        phase_d_summary={"backbone": {"ready_backbones": ["ridge", "tft"]}},
        social_throughput={},
    )
    assert abs(float(out["text_coverage_ratio"]) - 0.25) < 1e-12
    assert abs(float(out["text_contribution_abs"]) - 0.08) < 1e-12
    assert abs(float(out["gate_openness"]) - 1.0) < 1e-12
    assert out["ready_backbones"] == ["ridge", "tft"]
    assert int(out["ready_backbone_count"]) == 2
