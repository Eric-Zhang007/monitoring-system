from __future__ import annotations

import pytest

from security_config import (
    DEFAULT_CORS_ALLOW_ORIGINS,
    _parse_origin_allowlist,
    build_cors_settings,
)


def test_parse_origin_allowlist_trims_and_dedupes():
    out = _parse_origin_allowlist(" http://a.com,http://b.com,http://a.com,, ")
    assert out == ["http://a.com", "http://b.com"]


def test_build_cors_settings_default_allowlist(monkeypatch):
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    monkeypatch.delenv("CORS_ALLOW_CREDENTIALS", raising=False)
    out = build_cors_settings()
    assert out["allow_credentials"] is True
    assert out["allow_origins"] == _parse_origin_allowlist(DEFAULT_CORS_ALLOW_ORIGINS)


def test_build_cors_settings_rejects_wildcard_with_credentials(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*,http://localhost:3001")
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "1")
    with pytest.raises(ValueError):
        build_cors_settings()


def test_build_cors_settings_allows_wildcard_without_credentials(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "0")
    out = build_cors_settings()
    assert out["allow_credentials"] is False
    assert out["allow_origins"] == ["*"]
