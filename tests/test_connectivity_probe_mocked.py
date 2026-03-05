from __future__ import annotations

from typing import Any, Dict, List

import requests

from backend.connectivity_service import ConnectivityService


class _Repo:
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def save_venue_connectivity_status(self, **kwargs):
        self.rows.append(dict(kwargs))
        return len(self.rows)

    def get_proxy_profile_secret_row(self, profile_id: str):
        _ = profile_id
        return None

    def resolve_proxy_profile_for_target(self, process_id=None, account_id=None):  # noqa: ANN001
        _ = (process_id, account_id)
        return None


class _Resp:
    def __init__(self, status: int):
        self.status_code = status
        self.headers = {}


def test_connectivity_probe_mocked_ok(monkeypatch):
    repo = _Repo()
    svc = ConnectivityService(repo=repo)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _Resp(200))
    async def _ok(*args, **kwargs):  # noqa: ANN001
        return True
    monkeypatch.setattr(svc, "_probe_ws", _ok)
    out = svc.probe(venue="bitget", persist=True)
    assert out["rest_ok"] is True
    assert out["ws_ok"] is True
    assert len(repo.rows) == 1


def test_connectivity_probe_mocked_timeout_and_403(monkeypatch):
    repo = _Repo()
    svc = ConnectivityService(repo=repo)

    def _raise(*args, **kwargs):  # noqa: ANN001
        raise requests.Timeout("timeout")

    monkeypatch.setattr(requests, "get", _raise)
    async def _ws_fail(*args, **kwargs):  # noqa: ANN001
        return False
    monkeypatch.setattr(svc, "_probe_ws", _ws_fail)
    out = svc.probe(venue="bitget", persist=True)
    assert out["rest_ok"] is False
    assert out["ws_ok"] is False
    assert "rest_error" in str(out["error"])

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _Resp(403))
    out2 = svc.probe(venue="bitget", persist=False)
    assert out2["rest_ok"] is False
    assert "rest_status:403" in str(out2["error"])
