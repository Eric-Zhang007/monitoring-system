from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "scripts"))

from validate_no_leakage import _evaluate_candidate_gate_check  # noqa: E402
from validate_no_leakage import _evaluate_multimodal_gate_snapshot_check  # noqa: E402


def test_candidate_gate_check_enforces_passed_and_freshness(tmp_path):
    registry = tmp_path / "candidate_registry.jsonl"
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat().replace("+00:00", "Z")
    registry.write_text(
        json.dumps(
            {
                "registered_at": old_ts,
                "gate": {"passed": False, "reasons": ["oos_mse_above_threshold"]},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = _evaluate_candidate_gate_check(
        registry_path=str(registry),
        require_passed=True,
        max_age_hours=48.0,
        now=datetime.now(timezone.utc),
    )
    assert out["required"] is True
    assert out["exists"] is True
    assert out["gate_passed"] is False
    assert out["stale"] is True
    assert out["effective_passed"] is False
    assert "candidate_gate_not_passed" in out["effective_reasons"]
    assert "candidate_registry_stale" in out["effective_reasons"]


def test_multimodal_gate_snapshot_check_validates_track_and_status(tmp_path):
    snapshot = tmp_path / "multimodal_gate_state.json"
    snapshot.write_text(
        json.dumps(
            {
                "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "track": "liquid",
                "passed": True,
                "reasons": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = _evaluate_multimodal_gate_snapshot_check(
        snapshot_path=str(snapshot),
        track="liquid",
        require_passed=True,
        max_age_hours=24.0,
        now=datetime.now(timezone.utc),
    )
    assert out["required"] is True
    assert out["exists"] is True
    assert out["track_match"] is True
    assert out["passed"] is True
    assert out["effective_passed"] is True
