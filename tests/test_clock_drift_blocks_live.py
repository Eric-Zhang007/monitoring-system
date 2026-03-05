from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest
from fastapi import HTTPException
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
    def start(self, **kwargs):
        return {"process_id": "p1", "status": "running", **kwargs}


class _ConnSvcOK:
    def probe(self, **kwargs):
        _ = kwargs
        return {"rest_ok": True, "ws_ok": True, "error": None}

    def _resolve_proxy_context(self, **kwargs):
        _ = kwargs
        return {"profile": None, "proxy_env": {}}


class _ClockSvcRed:
    def probe(self, **kwargs):
        _ = kwargs
        return {"level": "red", "drift_ms": 99999}


def test_clock_drift_blocks_live(monkeypatch):
    monkeypatch.setenv("RBAC_ENFORCE", "1")
    monkeypatch.setattr(router_mod, "_PROCESS_MANAGER", _PM(), raising=False)
    monkeypatch.setattr(router_mod, "_CONNECTIVITY_SERVICE", _ConnSvcOK(), raising=False)
    monkeypatch.setattr(router_mod, "_CLOCK_DRIFT_SERVICE", _ClockSvcRed(), raising=False)

    payload: Dict[str, Any] = {"task_type": "LIVE_TRADER", "params": {"venue": "bitget"}}
    with pytest.raises(HTTPException) as ex:
        asyncio.run(router_mod.ops_process_start(_req("operator"), payload))
    assert ex.value.status_code == 503
    assert "clock_drift_red" in str(ex.value.detail)
