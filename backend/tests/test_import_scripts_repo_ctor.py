from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_backend_modules(monkeypatch, captured: dict):
    schemas_mod = ModuleType("schemas_v2")

    class FakeEvent:
        def __init__(self, **kwargs):
            self.payload = kwargs

    schemas_mod.Event = FakeEvent

    repo_mod = ModuleType("v2_repository")

    class FakeRepo:
        def __init__(self, *args, **kwargs):
            captured["repo_init_args"] = args
            captured["repo_init_kwargs"] = kwargs

        def ingest_events(self, events):
            n = len(events)
            return n, n, 0, list(range(1, n + 1))

    repo_mod.V2Repository = FakeRepo

    monkeypatch.setitem(sys.modules, "schemas_v2", schemas_mod)
    monkeypatch.setitem(sys.modules, "v2_repository", repo_mod)


def test_import_events_script_uses_db_url_constructor_kwarg(monkeypatch, tmp_path):
    mod = _load_script_module("import_events_jsonl.py")
    captured: dict = {}
    _install_fake_backend_modules(monkeypatch, captured)

    jsonl = tmp_path / "events.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "event_type": "market",
                "title": "t",
                "occurred_at": "2026-01-01T00:00:00Z",
                "payload": {},
                "entities": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["import_events_jsonl.py", "--jsonl", str(jsonl), "--database-url", "postgresql://x"])
    rc = mod.main()
    assert rc == 0
    assert captured.get("repo_init_kwargs", {}).get("db_url") == "postgresql://x"
    assert "dsn" not in captured.get("repo_init_kwargs", {})


def test_import_social_script_uses_db_url_constructor_kwarg(monkeypatch, tmp_path):
    mod = _load_script_module("import_social_events_jsonl.py")
    captured: dict = {}
    _install_fake_backend_modules(monkeypatch, captured)

    jsonl = tmp_path / "social.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "platform": "reddit",
                "title": "btc post",
                "content": "btc",
                "created_at": 1735689600,
                "source_url": "https://example.com/x",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["import_social_events_jsonl.py", "--jsonl", str(jsonl), "--database-url", "postgresql://y"])
    rc = mod.main()
    assert rc == 0
    assert captured.get("repo_init_kwargs", {}).get("db_url") == "postgresql://y"
    assert "dsn" not in captured.get("repo_init_kwargs", {})
