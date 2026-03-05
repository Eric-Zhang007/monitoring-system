from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from training.universe.top50 import UniverseBuildConfig, build_top_universe_snapshot


def test_universe_snapshot_hash_changes_when_rank_by_changes(monkeypatch, tmp_path: Path):
    ranked_rows = [
        {"symbol": "BTC", "volume_usd_30d": 10_000_000.0, "notional_usd_30d": 10_000_000.0, "adv_usd_30d": 333_333.0, "bars": 1000, "active_days": 30},
        {"symbol": "ETH", "volume_usd_30d": 8_000_000.0, "notional_usd_30d": 8_000_000.0, "adv_usd_30d": 266_666.0, "bars": 1000, "active_days": 30},
        {"symbol": "SOL", "volume_usd_30d": 6_000_000.0, "notional_usd_30d": 6_000_000.0, "adv_usd_30d": 200_000.0, "bars": 1000, "active_days": 30},
    ]

    monkeypatch.setattr(
        "training.universe.top50._rank_symbols_by_liquidity",
        lambda **kwargs: list(ranked_rows),  # noqa: ARG005
    )

    cfg_v = UniverseBuildConfig(
        as_of=datetime(2026, 3, 1, tzinfo=timezone.utc),
        top_n=2,
        lookback_days=30,
        timeframe="1h",
        min_notional_usd=1.0,
        exclude_stable=True,
        exclude_leveraged=True,
        hysteresis_keep_rank=3,
        track="liquid",
        rank_by="volume_usd_30d",
        source="db",
    )
    snap_v = build_top_universe_snapshot(database_url="postgresql://unused", cfg=cfg_v, previous_snapshot_file=tmp_path / "none.json")

    cfg_a = UniverseBuildConfig(
        as_of=cfg_v.as_of,
        top_n=2,
        lookback_days=30,
        timeframe="1h",
        min_notional_usd=1.0,
        exclude_stable=True,
        exclude_leveraged=True,
        hysteresis_keep_rank=3,
        track="liquid",
        rank_by="adv_usd_30d",
        source="db",
    )
    snap_a = build_top_universe_snapshot(database_url="postgresql://unused", cfg=cfg_a, previous_snapshot_file=tmp_path / "none.json")
    assert snap_v["snapshot_hash"] != snap_a["snapshot_hash"]

