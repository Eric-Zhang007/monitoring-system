from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "build_multisource_events_2025.py"
    spec = importlib.util.spec_from_file_location("build_multisource_events_2025", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_event(provider: str, idx: int):
    ts = f"2025-03-{(idx % 9) + 1:02d}T12:{(idx % 60):02d}:00Z"
    return {
        "title": f"{provider}-{idx}",
        "occurred_at": ts,
        "source_url": f"https://example.com/{provider}/{idx}",
        "event_importance": 0.6,
        "confidence_score": 0.65,
        "novelty_score": 0.5,
        "payload": {"provider": provider},
    }


def test_latency_model_is_positive_and_source_sensitive():
    mod = _load_module()
    now = datetime.now(timezone.utc)
    lag_google = mod._estimate_source_latency_minutes(
        provider="google_news_rss",
        source_name="example",
        title="btc rally",
        link="https://x/a",
        occurred_at=now,
        source_tier=2,
    )
    lag_official = mod._estimate_source_latency_minutes(
        provider="rss:federal_reserve",
        source_name="federal_reserve",
        title="fomc statement",
        link="https://x/b",
        occurred_at=now,
        source_tier=1,
    )
    assert lag_google > 0
    assert lag_official > 0
    assert lag_google > lag_official


def test_cap_provider_share_reduces_google_dominance():
    mod = _load_module()
    events = [_fake_event("google_news_rss", i) for i in range(90)] + [_fake_event("gdelt", i) for i in range(10)]
    balanced, meta = mod._cap_provider_share(events, provider="google_news_rss", max_share=0.65, min_keep=20)
    providers = [mod._provider_of(ev) for ev in balanced]
    google_n = sum(1 for p in providers if p == "google_news_rss")
    share = google_n / max(1, len(balanced))
    assert meta["status"] in {"capped", "noop_under_cap"}
    expected_max = max(0.65, 20.0 / (20.0 + 10.0))
    assert share <= expected_max + 1e-6
    assert google_n >= 20
