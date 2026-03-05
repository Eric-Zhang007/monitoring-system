from __future__ import annotations

from typing import Any, Dict, List

import requests

from backend.connectivity_service import ConnectivityService


class _Repo:
    def __init__(self):
        self.saved: List[Dict[str, Any]] = []

    def get_proxy_profile_secret_row(self, profile_id: str):
        return {
            "profile_id": profile_id,
            "proxy_type": "http",
            "host": "127.0.0.1",
            "port": 7890,
            "username": "u",
            "password_enc": "enc",
            "enabled": True,
        }

    def resolve_proxy_profile_for_target(self, process_id=None, account_id=None):  # noqa: ANN001
        _ = (process_id, account_id)
        return None

    def save_venue_connectivity_status(self, **kwargs):
        self.saved.append(dict(kwargs))
        return len(self.saved)


class _Secrets:
    def decrypt(self, text: str) -> str:
        _ = text
        return "p"


class _Resp:
    def __init__(self, status: int):
        self.status_code = status
        self.headers = {}


def test_connectivity_probe_with_proxy_profile(monkeypatch):
    repo = _Repo()
    svc = ConnectivityService(repo=repo, secrets_manager=_Secrets())
    seen: Dict[str, Any] = {}

    def _fake_get(*args, **kwargs):  # noqa: ANN001
        seen["proxies"] = kwargs.get("proxies")
        return _Resp(200)

    monkeypatch.setattr(requests, "get", _fake_get)
    async def _ok(*args, **kwargs):  # noqa: ANN001
        return True
    monkeypatch.setattr(svc, "_probe_ws", _ok)
    out = svc.probe(venue="bitget", proxy_profile_id="px1", persist=True)
    assert out["using_proxy_profile"] == "px1"
    assert isinstance(seen.get("proxies"), dict)
    assert "http" in seen["proxies"]
    assert len(repo.saved) == 1
