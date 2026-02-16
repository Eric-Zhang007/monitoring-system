from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "orchestrate_event_social_backfill.py"
    spec = importlib.util.spec_from_file_location("orchestrate_event_social_backfill", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_event_time_alignment_enforces_monotonic():
    mod = _load_module()
    fallback_now = datetime(2026, 2, 16, tzinfo=timezone.utc)
    ev = {
        "title": "x",
        "occurred_at": "2026-01-01T12:00:00Z",
        "published_at": "2025-12-31T00:00:00Z",
        "available_at": "2025-12-30T00:00:00Z",
        "effective_at": "2025-12-29T00:00:00Z",
        "payload": {},
    }
    out, monotonic = mod._normalize_event_time_alignment(ev, fallback_now=fallback_now)
    assert monotonic is True
    assert out["occurred_at"] <= out["published_at"] <= out["available_at"] <= out["effective_at"]
    assert int(out["source_latency_ms"]) >= 0
    assert int(out["latency_ms"]) >= 0


def test_enrich_chunk_file_adds_provenance_and_filters_languages(tmp_path):
    mod = _load_module()
    in_path = tmp_path / "events.jsonl"
    rows = [
        {
            "event_type": "market",
            "title": "btc rally",
            "occurred_at": "2026-01-01T00:00:00Z",
            "published_at": "2026-01-01T00:00:00Z",
            "available_at": "2026-01-01T00:05:00Z",
            "effective_at": "2026-01-01T00:05:00Z",
            "source_url": "https://example.com/a",
            "payload": {"language": "en"},
        },
        {
            "event_type": "market",
            "title": "filtered item",
            "occurred_at": "2026-01-01T00:00:00Z",
            "published_at": "2026-01-01T00:00:00Z",
            "available_at": "2026-01-01T00:01:00Z",
            "effective_at": "2026-01-01T00:01:00Z",
            "source_url": "https://example.com/b",
            "payload": {"language": "fr"},
        },
    ]
    with open(in_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    stats = mod._enrich_chunk_file(
        path=str(in_path),
        stream="events",
        chunk_start=datetime(2025, 12, 31, tzinfo=timezone.utc),
        chunk_end=datetime(2026, 1, 2, tzinfo=timezone.utc),
        pipeline_tag="task_h",
        run_id="run-1",
        language_targets=["en", "zh"],
        google_locales=["US:en", "CN:zh-Hans"],
    )
    assert stats["input_rows"] == 2
    assert stats["output_rows"] == 1
    assert stats["dropped_language"] == 1

    out_rows = mod._read_jsonl(str(in_path))
    assert len(out_rows) == 1
    payload = out_rows[0]["payload"]
    assert payload["provenance"]["pipeline_tag"] == "task_h"
    assert payload["provenance"]["orchestrator_run_id"] == "run-1"
    assert payload["time_alignment"]["alignment_mode"] == "strict_asof_v1"
