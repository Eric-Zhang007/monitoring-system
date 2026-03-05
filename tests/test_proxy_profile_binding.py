from __future__ import annotations

import asyncio
from typing import Any, Dict

from starlette.requests import Request

import v2_router as router_mod


def _req(role: str = "operator") -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(b"x-role", role.encode("utf-8")), (b"x-user", b"tester")],
        }
    )


class _PM:
    def __init__(self):
        self.last_params: Dict[str, Any] = {}

    def start(self, *, task_type: str, params: Dict[str, Any], created_by: str):
        self.last_params = dict(params)
        return {"process_id": "p1", "status": "running", "task_type": task_type, "created_by": created_by, "env_overrides": dict(params.get("env_overrides") or {})}


class _ConnSvc:
    def probe(self, **kwargs):
        _ = kwargs
        return {"rest_ok": True, "ws_ok": True, "error": None}

    def _resolve_proxy_context(self, **kwargs):
        _ = kwargs
        return {
            "profile": {"profile_id": "px1"},
            "proxy_env": {"HTTP_PROXY": "http://127.0.0.1:7890", "HTTPS_PROXY": "http://127.0.0.1:7890"},
        }


class _ClockSvc:
    def probe(self, **kwargs):
        _ = kwargs
        return {"level": "green", "drift_ms": 10}


def test_proxy_profile_binding_applies_env(monkeypatch):
    monkeypatch.setenv("RBAC_ENFORCE", "1")
    pm = _PM()
    monkeypatch.setattr(router_mod, "_PROCESS_MANAGER", pm, raising=False)
    monkeypatch.setattr(router_mod, "_CONNECTIVITY_SERVICE", _ConnSvc(), raising=False)
    monkeypatch.setattr(router_mod, "_CLOCK_DRIFT_SERVICE", _ClockSvc(), raising=False)

    payload = {
        "task_type": "LIVE_TRADER",
        "params": {"venue": "bitget", "account_id": "acc1"},
    }
    out = asyncio.run(router_mod.ops_process_start(_req("operator"), payload))
    envs = out["process"]["env_overrides"]
    assert envs.get("HTTP_PROXY") == "http://127.0.0.1:7890"
    assert out["process"].get("status") == "running"
