from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import v2_router as router_mod


def _req(role: str) -> Request:
    return Request({"type": "http", "headers": [(b"x-role", role.encode("utf-8")), (b"x-user", b"tester")]})


def test_rbac_blocks_viewer_from_process_start(monkeypatch):
    monkeypatch.setenv("RBAC_ENFORCE", "1")
    monkeypatch.setattr(router_mod, "_PROCESS_MANAGER", object(), raising=False)
    with pytest.raises(HTTPException) as ex:
        asyncio.run(router_mod.ops_process_start(_req("viewer"), {"task_type": "TRAIN_LIQUID", "params": {}}))
    assert ex.value.status_code == 403


def test_rbac_blocks_operator_from_admin_account_write(monkeypatch):
    monkeypatch.setenv("RBAC_ENFORCE", "1")

    class _Secrets:
        def encrypt_bitget(self, **kwargs):
            return {"api_key_enc": "a", "api_secret_enc": "b", "passphrase_enc": "c"}

    monkeypatch.setattr(router_mod, "_SECRETS_MANAGER", _Secrets(), raising=False)
    with pytest.raises(HTTPException) as ex:
        asyncio.run(
            router_mod.upsert_live_account(
                _req("operator"),
                {
                    "account_id": "acc1",
                    "account_name": "acc1",
                    "api_key": "k",
                    "api_secret": "s",
                    "passphrase": "p",
                },
            )
        )
    assert ex.value.status_code == 403
