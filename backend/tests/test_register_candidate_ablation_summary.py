from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "training"))

from register_candidate_model import build_ablation_summary  # noqa: E402
from register_candidate_model import evaluate_candidate_gate  # noqa: E402


def test_build_ablation_summary_extracts_deltas():
    report = {
        "primary_ablation": "full",
        "ablation_results": [
            {"ablation": "full", "status": "ok", "mse": 0.10, "hit_rate": 0.55, "rows": 1000},
            {"ablation": "no_text", "status": "ok", "mse": 0.12, "hit_rate": 0.53, "rows": 1000},
            {"ablation": "no_macro", "status": "ok", "mse": 0.11, "hit_rate": 0.54, "rows": 1000},
            {"ablation": "event_window", "status": "ok", "mse": 0.08, "hit_rate": 0.58, "rows": 220},
        ],
    }
    out = build_ablation_summary(report)
    assert out["available"] is True
    assert out["count"] == 4
    assert abs(float(out["delta_mse_no_text_vs_full"]) - 0.02) < 1e-12
    assert abs(float(out["delta_hit_event_window_vs_full"]) - 0.03) < 1e-12


def test_evaluate_candidate_gate_checks_backbones_and_oos():
    row = {
        "oos_hit_rate": 0.54,
        "oos_mse": 0.12,
        "backbone_ready_list": ["ridge", "itransformer"],
        "ablation_summary": {
            "available": True,
            "delta_mse_no_text_vs_full": 0.02,
            "delta_mse_no_macro_vs_full": 0.01,
        },
    }
    passed, reasons, metrics = evaluate_candidate_gate(
        row,
        min_oos_hit_rate=0.53,
        max_oos_mse=0.2,
        min_backbone_ready=2,
        required_backbones=["ridge"],
        max_delta_mse_no_text_vs_full=0.05,
        max_delta_mse_no_macro_vs_full=0.05,
    )
    assert passed is True
    assert reasons == []
    assert int(metrics["ready_backbone_count"]) == 2


def test_evaluate_candidate_gate_fails_when_thresholds_violated():
    row = {
        "oos_hit_rate": 0.49,
        "oos_mse": 0.22,
        "backbone_ready_list": ["ridge"],
        "ablation_summary": {
            "available": True,
            "delta_mse_no_text_vs_full": 0.12,
            "delta_mse_no_macro_vs_full": 0.01,
        },
    }
    passed, reasons, _ = evaluate_candidate_gate(
        row,
        min_oos_hit_rate=0.50,
        max_oos_mse=0.2,
        min_backbone_ready=2,
        required_backbones=["ridge", "patchtst"],
        max_delta_mse_no_text_vs_full=0.05,
        max_delta_mse_no_macro_vs_full=0.05,
    )
    assert passed is False
    assert "oos_hit_rate_below_threshold" in reasons
    assert "oos_mse_above_threshold" in reasons
    assert "ready_backbone_count_below_threshold" in reasons
    assert "required_backbones_not_ready" in reasons
    assert "delta_mse_no_text_vs_full_above_threshold" in reasons
