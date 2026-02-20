from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "import_social_raw_jsonl.py"
    spec = importlib.util.spec_from_file_location("import_social_raw_jsonl", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_to_raw_rows_converts_posts_and_synthetic_comments():
    mod = _load_module()
    rows = [
        {
            "title": "BTC rally",
            "occurred_at": "2026-02-01T00:00:00Z",
            "available_at": "2026-02-01T00:05:00Z",
            "source_url": "https://example.com/a",
            "payload": {
                "social_platform": "reddit",
                "summary": "bullish",
                "symbol_mentions": ["BTC"],
                "engagement_score": 12.0,
                "post_sentiment": 0.7,
                "comment_sentiment": 0.2,
                "n_comments": 3,
                "n_replies": 1,
                "author": "alice",
            },
        }
    ]
    posts, comments = mod._to_raw_rows(rows)
    assert len(posts) == 1
    assert len(comments) == 1
    assert posts[0][0] == "reddit"
    assert "BTC" in posts[0][14]
    assert comments[0][2] == posts[0][1]  # parent_source_id links to post source_id


def test_main_dry_run_writes_artifact(tmp_path, monkeypatch):
    mod = _load_module()
    jsonl = tmp_path / "social.jsonl"
    out_json = tmp_path / "social_raw_import.json"
    jsonl.write_text(
        json.dumps(
            {
                "title": "ETH update",
                "occurred_at": "2026-02-01T01:00:00Z",
                "available_at": "2026-02-01T01:01:00Z",
                "payload": {
                    "social_platform": "x",
                    "summary": "neutral",
                    "symbol_mentions": ["ETH"],
                    "n_comments": 0,
                    "n_replies": 0,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_social_raw_jsonl.py",
            "--jsonl",
            str(jsonl),
            "--dry-run",
            "--out-json",
            str(out_json),
        ],
    )
    rc = mod.main()
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["dry_run"] is True
    assert int(payload["posts_converted"]) == 1
