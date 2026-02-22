from __future__ import annotations

import asyncio

from nim_integration import NIMFeatureCache


def test_nim_failure_returns_missing(monkeypatch):
    cache = object.__new__(NIMFeatureCache)

    monkeypatch.setenv("NIM_EMBED_ENDPOINT", "http://x")
    monkeypatch.setattr("nim_integration.requests.post", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    out = asyncio.run(NIMFeatureCache._call_nim_embedding(cache, "hello"))
    assert out is None
