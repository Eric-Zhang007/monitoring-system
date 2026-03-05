from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import v2_router as router_mod


def _req(role: str = "operator", user: str = "tester") -> Request:
    scope = {
        "type": "http",
        "headers": [
            (b"x-role", role.encode("utf-8")),
            (b"x-user", user.encode("utf-8")),
        ],
    }
    return Request(scope)


class _PM:
    def __init__(self):
        self.called = False

    def start(self, **kwargs):
        self.called = True
        return {"process_id": "p1", "status": "running", **kwargs}


class _ConnSvcFail:
    def probe(self, **kwargs):
        _ = kwargs
        return {"rest_ok": False, "ws_ok": False, "error": "network_down"}


class _ClockSvc:
    def probe(self, **kwargs):
        _ = kwargs
        return {"level": "green", "drift_ms": 10}


def test_live_start_blocked_when_unreachable(monkeypatch):
    monkeypatch.setenv("RBAC_ENFORCE", "1")
    pm = _PM()
    monkeypatch.setattr(router_mod, "_PROCESS_MANAGER", pm, raising=False)
    monkeypatch.setattr(router_mod, "_CONNECTIVITY_SERVICE", _ConnSvcFail(), raising=False)
    monkeypatch.setattr(router_mod, "_CLOCK_DRIFT_SERVICE", _ClockSvc(), raising=False)

    payload: Dict[str, Any] = {"task_type": "LIVE_TRADER", "params": {"venue": "bitget"}}
    with pytest.raises(HTTPException) as ex:
        asyncio.run(router_mod.ops_process_start(_req("operator"), payload))
    assert ex.value.status_code == 503
    assert "network_unreachable_for_live" in str(ex.value.detail)
    assert pm.called is False
