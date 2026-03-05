from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from v2_repository import V2Repository


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class RuntimeConfigItem:
    config_key: str
    scope: str
    scope_id: str
    value_json: Dict[str, Any]
    version: int
    requires_restart: bool
    description: str
    updated_by: str
    updated_at: str


class RuntimeConfigService:
    def __init__(self, repo: V2Repository):
        self.repo = repo
        self._cache: Dict[str, RuntimeConfigItem] = {}
        self._cache_loaded_at: str = ""

    @staticmethod
    def _cache_key(config_key: str, scope: str, scope_id: str) -> str:
        return f"{scope}:{scope_id}:{config_key}"

    def refresh_cache(self) -> Dict[str, Any]:
        rows = self.repo.list_runtime_config()
        next_cache: Dict[str, RuntimeConfigItem] = {}
        for r in rows:
            item = RuntimeConfigItem(
                config_key=str(r.get("config_key") or ""),
                scope=str(r.get("scope") or "global"),
                scope_id=str(r.get("scope_id") or ""),
                value_json=dict(r.get("value_json") or {}),
                version=int(r.get("version") or 1),
                requires_restart=bool(r.get("requires_restart") or False),
                description=str(r.get("description") or ""),
                updated_by=str(r.get("updated_by") or "system"),
                updated_at=(r.get("updated_at").astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if r.get("updated_at") else _utcnow_iso()),
            )
            key = self._cache_key(item.config_key, item.scope, item.scope_id)
            next_cache[key] = item
        self._cache = next_cache
        self._cache_loaded_at = _utcnow_iso()
        return {"items": len(next_cache), "loaded_at": self._cache_loaded_at}

    def list_config(self, *, scope: Optional[str] = None, scope_id: Optional[str] = None, config_key: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = self.repo.list_runtime_config(scope=scope, scope_id=scope_id, config_key=config_key)
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "config_key": str(r.get("config_key") or ""),
                    "scope": str(r.get("scope") or "global"),
                    "scope_id": str(r.get("scope_id") or ""),
                    "value_json": dict(r.get("value_json") or {}),
                    "version": int(r.get("version") or 1),
                    "requires_restart": bool(r.get("requires_restart") or False),
                    "description": str(r.get("description") or ""),
                    "updated_by": str(r.get("updated_by") or "system"),
                    "updated_at": (r.get("updated_at").astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if r.get("updated_at") else _utcnow_iso()),
                }
            )
        return out

    def upsert_config(
        self,
        *,
        config_key: str,
        value_json: Dict[str, Any],
        scope: str = "global",
        scope_id: str = "",
        requires_restart: bool = False,
        description: str = "",
        updated_by: str = "system",
    ) -> Dict[str, Any]:
        row = self.repo.upsert_runtime_config(
            config_key=str(config_key),
            scope=str(scope),
            scope_id=str(scope_id),
            value_json=dict(value_json or {}),
            requires_restart=bool(requires_restart),
            description=str(description or ""),
            updated_by=str(updated_by or "system"),
        )
        self.refresh_cache()
        return {
            "config_key": str(row.get("config_key") or config_key),
            "scope": str(row.get("scope") or scope),
            "scope_id": str(row.get("scope_id") or scope_id),
            "value_json": dict(row.get("value_json") or value_json or {}),
            "version": int(row.get("version") or 1),
            "requires_restart": bool(row.get("requires_restart") or requires_restart),
            "description": str(row.get("description") or description or ""),
            "updated_by": str(row.get("updated_by") or updated_by),
            "updated_at": (row.get("updated_at").astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if row.get("updated_at") else _utcnow_iso()),
        }

    def list_audit_logs(self, *, limit: int = 200, scope: Optional[str] = None, scope_id: Optional[str] = None, config_key: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = self.repo.list_runtime_config_audit_logs(limit=limit, scope=scope, scope_id=scope_id, config_key=config_key)
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r.get("id") or 0),
                    "config_key": str(r.get("config_key") or ""),
                    "scope": str(r.get("scope") or "global"),
                    "scope_id": str(r.get("scope_id") or ""),
                    "old_value_json": dict(r.get("old_value_json") or {}),
                    "new_value_json": dict(r.get("new_value_json") or {}),
                    "old_version": int(r.get("old_version") or 0) if r.get("old_version") is not None else None,
                    "new_version": int(r.get("new_version") or 0) if r.get("new_version") is not None else None,
                    "requires_restart": bool(r.get("requires_restart") or False),
                    "updated_by": str(r.get("updated_by") or "system"),
                    "created_at": (r.get("created_at").astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if r.get("created_at") else _utcnow_iso()),
                }
            )
        return out

    def resolve(
        self,
        *,
        config_key: str,
        scope: str = "global",
        scope_id: str = "",
        env_default: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._cache:
            self.refresh_cache()
        keys = [
            self._cache_key(config_key, scope, scope_id),
            self._cache_key(config_key, scope, ""),
            self._cache_key(config_key, "global", ""),
        ]
        for k in keys:
            row = self._cache.get(k)
            if row is not None:
                return {
                    "found": True,
                    "config_key": row.config_key,
                    "scope": row.scope,
                    "scope_id": row.scope_id,
                    "value_json": dict(row.value_json),
                    "version": row.version,
                    "requires_restart": row.requires_restart,
                    "description": row.description,
                    "updated_by": row.updated_by,
                    "updated_at": row.updated_at,
                    "source": "runtime_config",
                }
        if env_default is None:
            env_default = os.getenv(str(config_key).upper(), "")
        return {
            "found": False,
            "config_key": str(config_key),
            "scope": str(scope),
            "scope_id": str(scope_id),
            "value_json": {"value": env_default},
            "version": 0,
            "requires_restart": False,
            "description": "env_default",
            "updated_by": "env",
            "updated_at": _utcnow_iso(),
            "source": "env",
        }
