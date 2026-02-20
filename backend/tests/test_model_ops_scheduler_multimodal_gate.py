from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import monitoring.model_ops_scheduler as scheduler  # noqa: E402


def test_multimodal_gate_writes_pass_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "multimodal_gate_state.json"
    audits = []

    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_SNAPSHOT", str(snapshot), raising=False)
    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_REQUIRE_PASSED", True, raising=False)
    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_MIN_READY_BACKBONES", 2, raising=False)
    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_REQUIRED_BACKBONES", ["ridge", "tft"], raising=False)
    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_MIN_TEXT_COVERAGE", 0.30, raising=False)
    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_MAX_DELTA_MSE_NO_TEXT", 0.20, raising=False)
    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_MAX_DELTA_MSE_NO_MACRO", 0.20, raising=False)
    monkeypatch.setattr(scheduler, "_audit_log", lambda **kw: audits.append(kw))

    def _fake_get(path: str, params=None):
        assert path == "/api/v2/monitor/model-status"
        return {
            "latest_candidate": {
                "gate": {"passed": True},
                "ablation_summary": {
                    "delta_mse_no_text_vs_full": 0.05,
                    "delta_mse_no_macro_vs_full": 0.04,
                },
            },
            "multimodal_health": {
                "candidate_gate_passed": True,
                "ready_backbones": ["ridge", "tft"],
                "text_coverage_ratio": 0.42,
            },
        }

    monkeypatch.setattr(scheduler, "_get", _fake_get)
    scheduler._run_multimodal_gate("liquid")

    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert payload["track"] == "liquid"
    assert payload["passed"] is True
    assert payload["reasons"] == []
    assert payload["metrics"]["ready_backbone_count"] == 2
    assert len(audits) == 1
    assert audits[0]["action"] == "multimodal_gate"


def test_multimodal_gate_records_failure_reasons(tmp_path, monkeypatch):
    snapshot = tmp_path / "multimodal_gate_state.json"

    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_SNAPSHOT", str(snapshot), raising=False)
    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_REQUIRE_PASSED", True, raising=False)
    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_MIN_READY_BACKBONES", 2, raising=False)
    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_REQUIRED_BACKBONES", ["ridge", "patchtst"], raising=False)
    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_MIN_TEXT_COVERAGE", 0.50, raising=False)
    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_MAX_DELTA_MSE_NO_TEXT", 0.05, raising=False)
    monkeypatch.setattr(scheduler, "MULTIMODAL_GATE_MAX_DELTA_MSE_NO_MACRO", 0.05, raising=False)
    monkeypatch.setattr(scheduler, "_audit_log", lambda **kwargs: None)

    def _fake_get(path: str, params=None):
        assert path == "/api/v2/monitor/model-status"
        return {
            "latest_candidate": {
                "gate": {"passed": False},
                "ablation_summary": {
                    "delta_mse_no_text_vs_full": 0.20,
                    "delta_mse_no_macro_vs_full": 0.10,
                },
            },
            "multimodal_health": {
                "candidate_gate_passed": False,
                "ready_backbones": ["ridge"],
                "text_coverage_ratio": 0.10,
            },
        }

    monkeypatch.setattr(scheduler, "_get", _fake_get)
    scheduler._run_multimodal_gate("liquid")

    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    reasons = set(payload.get("reasons") or [])
    assert payload["passed"] is False
    assert "candidate_gate_not_passed" in reasons
    assert "ready_backbones_below_threshold" in reasons
    assert "required_backbones_not_ready" in reasons
    assert "text_coverage_below_threshold" in reasons
    assert "delta_mse_no_text_vs_full_above_threshold" in reasons
    assert "delta_mse_no_macro_vs_full_above_threshold" in reasons
