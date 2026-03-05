from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.runtime_config_service import RuntimeConfigService


class _FakeRepo:
    def __init__(self):
        self.rows: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        self.audit_logs: List[Dict[str, Any]] = []

    def list_runtime_config(self, *, scope: Optional[str] = None, scope_id: Optional[str] = None, config_key: Optional[str] = None):
        out = []
        for _, row in self.rows.items():
            if scope is not None and str(row["scope"]) != str(scope):
                continue
            if scope_id is not None and str(row["scope_id"]) != str(scope_id):
                continue
            if config_key is not None and str(row["config_key"]) != str(config_key):
                continue
            out.append(dict(row))
        return out

    def upsert_runtime_config(self, *, config_key: str, scope: str, scope_id: str, value_json: Dict[str, Any], requires_restart: bool, description: str, updated_by: str):
        key = (str(config_key), str(scope), str(scope_id))
        old = self.rows.get(key)
        version = int(old["version"]) + 1 if old else 1
        row = {
            "config_key": str(config_key),
            "scope": str(scope),
            "scope_id": str(scope_id),
            "value_json": dict(value_json),
            "version": version,
            "requires_restart": bool(requires_restart),
            "description": str(description),
            "updated_by": str(updated_by),
            "updated_at": None,
        }
        self.rows[key] = row
        self.audit_logs.append({"config_key": config_key, "scope": scope, "scope_id": scope_id, "new_version": version})
        return dict(row)

    def list_runtime_config_audit_logs(self, *, limit: int = 200, scope: Optional[str] = None, scope_id: Optional[str] = None, config_key: Optional[str] = None):
        out = []
        for x in reversed(self.audit_logs):
            if scope is not None and str(x.get("scope")) != str(scope):
                continue
            if scope_id is not None and str(x.get("scope_id")) != str(scope_id):
                continue
            if config_key is not None and str(x.get("config_key")) != str(config_key):
                continue
            out.append(dict(x))
            if len(out) >= limit:
                break
        return out


def test_runtime_config_service_hot_reload():
    repo = _FakeRepo()
    svc = RuntimeConfigService(repo)  # type: ignore[arg-type]
    first = svc.upsert_config(
        config_key="paper.initial_usdt",
        value_json={"value": 500},
        scope="global",
        scope_id="",
        requires_restart=False,
        description="paper balance",
        updated_by="pytest",
    )
    assert first["version"] == 1
    resolved = svc.resolve(config_key="paper.initial_usdt")
    assert resolved["found"] is True
    assert resolved["value_json"]["value"] == 500

    second = svc.upsert_config(
        config_key="paper.initial_usdt",
        value_json={"value": 650},
        scope="global",
        scope_id="",
        requires_restart=False,
        description="paper balance updated",
        updated_by="pytest2",
    )
    assert second["version"] == 2
    svc.refresh_cache()
    resolved2 = svc.resolve(config_key="paper.initial_usdt")
    assert resolved2["found"] is True
    assert resolved2["value_json"]["value"] == 650
