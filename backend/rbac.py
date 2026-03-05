from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

from fastapi import HTTPException, Request


ROLE_VIEWER = "viewer"
ROLE_OPERATOR = "operator"
ROLE_ADMIN = "admin"
VALID_ROLES = {ROLE_VIEWER, ROLE_OPERATOR, ROLE_ADMIN}

ROLE_LEVEL = {
    ROLE_VIEWER: 1,
    ROLE_OPERATOR: 2,
    ROLE_ADMIN: 3,
}


def resolve_actor_role(request: Request) -> Tuple[str, str]:
    actor = str(request.headers.get("x-user") or request.headers.get("x-actor") or "anonymous").strip() or "anonymous"
    role = str(request.headers.get("x-role") or request.headers.get("x-user-role") or "").strip().lower()
    if role not in VALID_ROLES:
        role = ROLE_ADMIN if os.getenv("RBAC_ENFORCE", "0").strip().lower() not in {"1", "true", "yes", "on"} else ROLE_VIEWER
    return actor, role


def require_role(request: Request, min_role: str) -> Tuple[str, str]:
    actor, role = resolve_actor_role(request)
    need = ROLE_LEVEL.get(str(min_role), 999)
    cur = ROLE_LEVEL.get(role, 0)
    if cur < need:
        raise HTTPException(status_code=403, detail=f"rbac_forbidden:need_{min_role}")
    return actor, role


def can(role: str, min_role: str) -> bool:
    return ROLE_LEVEL.get(role, 0) >= ROLE_LEVEL.get(min_role, 999)
