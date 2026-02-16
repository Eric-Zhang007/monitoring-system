from __future__ import annotations

import os
from typing import Dict, List


DEFAULT_CORS_ALLOW_ORIGINS = (
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://localhost:3001,"
    "http://127.0.0.1:3001"
)


def _parse_bool_env(value: str, *, default: bool) -> bool:
    raw = str(value or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _parse_origin_allowlist(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for part in str(raw or "").split(","):
        item = part.strip()
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_cors_settings() -> Dict[str, object]:
    origins_raw = os.getenv("CORS_ALLOW_ORIGINS", DEFAULT_CORS_ALLOW_ORIGINS)
    allow_origins = _parse_origin_allowlist(origins_raw)
    allow_credentials = _parse_bool_env(os.getenv("CORS_ALLOW_CREDENTIALS", "1"), default=True)

    if not allow_origins:
        raise ValueError("CORS_ALLOW_ORIGINS is empty; provide at least one explicit origin")
    if allow_credentials and "*" in allow_origins:
        raise ValueError("CORS_ALLOW_ORIGINS cannot include '*' when CORS_ALLOW_CREDENTIALS=1")

    return {
        "allow_origins": allow_origins,
        "allow_credentials": allow_credentials,
        "allow_methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        "allow_headers": ["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
    }
