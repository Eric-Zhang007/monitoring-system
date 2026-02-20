from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "backfill_social_history.py"
    spec = importlib.util.spec_from_file_location("backfill_social_history", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comment_backfill_marks_synthetic_fields():
    mod = _load_module()
    events = [
        {
            "title": "x",
            "payload": {
                "n_comments": 0,
                "n_replies": 0,
                "engagement_score": 10,
                "author_followers": 100,
                "post_sentiment": 0.3,
                "symbol_mentions": ["BTC"],
            },
        }
    ]
    out = mod._apply_comment_backfill(
        events,
        target_ratio=2.0,
        symbols=["BTC"],
        max_retry=1,
        page_limit=10,
    )
    assert int(out.get("comments_added") or 0) > 0
    payload = events[0]["payload"]
    assert int(payload.get("comment_backfill_added") or 0) > 0
    assert bool(payload.get("comment_backfill_synthetic")) is True
