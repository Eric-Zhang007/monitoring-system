from __future__ import annotations

import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402
from schemas_v2 import SchedulerAuditLogRequest  # noqa: E402


class _FakeRepo:
    def __init__(self):
        self.audit_logs = []
        self.events = []

    def get_model_rollout_state(self, track: str):
        return None

    def save_scheduler_audit_log(self, **kwargs):
        self.audit_logs.append(kwargs)
        return None

    def save_risk_event(self, **kwargs):
        self.events.append(kwargs)
        return None


def test_rollout_state_returns_default_when_missing(monkeypatch):
    fake = _FakeRepo()
    monkeypatch.setattr(router_mod, "repo", fake)
    resp = asyncio.run(router_mod.get_rollout_state("liquid"))
    assert resp.track == "liquid"
    assert resp.stage_pct == 10
    assert resp.status == "shadow"
    assert resp.model_name == "liquid_ttm_ensemble"


def test_scheduler_audit_log_is_persisted(monkeypatch):
    fake = _FakeRepo()
    monkeypatch.setattr(router_mod, "repo", fake)
    payload = SchedulerAuditLogRequest(
        track="liquid",
        action="rollout_advance",
        window={"hours": 24},
        thresholds={"min_hit_rate": 0.45},
        decision={"promoted": False, "reason": "hard_limit_failed"},
    )
    out = asyncio.run(router_mod.create_scheduler_audit_log(payload))
    assert out["status"] == "ok"
    assert len(fake.audit_logs) == 1
    assert fake.audit_logs[0]["track"] == "liquid"
    assert fake.audit_logs[0]["action"] == "rollout_advance"


def test_alert_notify_maps_severity_and_code(monkeypatch):
    fake = _FakeRepo()
    monkeypatch.setattr(router_mod, "repo", fake)
    payload = {
        "status": "firing",
        "alerts": [
            {
                "labels": {"alertname": "ExecutionRejectRateCritical", "severity": "P1"},
                "annotations": {"summary": "reject rate high"},
            },
            {
                "labels": {"alertname": "SignalLatencyP99Degraded", "severity": "P2"},
                "annotations": {"summary": "latency degraded"},
            },
        ],
    }
    out = asyncio.run(router_mod.ingest_alert_notification(payload))
    assert out["status"] == "ok"
    assert out["alerts_received"] == 2
    assert len(fake.events) == 2
    assert fake.events[0]["severity"] == "critical"
    assert fake.events[0]["code"] == "alertmanager:ExecutionRejectRateCritical"
    assert fake.events[1]["severity"] == "warning"
    assert fake.events[1]["code"] == "alertmanager:SignalLatencyP99Degraded"
