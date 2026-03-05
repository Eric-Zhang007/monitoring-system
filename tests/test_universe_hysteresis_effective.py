from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from training.universe.top50 import UniverseBuildConfig, build_top_universe_snapshot


def test_hysteresis_keeps_previous_member_inside_keep_rank(monkeypatch, tmp_path: Path):
    ranked = [
        {"symbol": "D", "volume_usd_30d": 600.0, "adv_usd_30d": 20.0, "notional_usd_30d": 600.0, "bars": 100, "active_days": 30},
        {"symbol": "A", "volume_usd_30d": 590.0, "adv_usd_30d": 19.0, "notional_usd_30d": 590.0, "bars": 100, "active_days": 30},
        {"symbol": "E", "volume_usd_30d": 580.0, "adv_usd_30d": 18.0, "notional_usd_30d": 580.0, "bars": 100, "active_days": 30},
        {"symbol": "B", "volume_usd_30d": 570.0, "adv_usd_30d": 17.0, "notional_usd_30d": 570.0, "bars": 100, "active_days": 30},
        {"symbol": "F", "volume_usd_30d": 560.0, "adv_usd_30d": 16.0, "notional_usd_30d": 560.0, "bars": 100, "active_days": 30},
        {"symbol": "C", "volume_usd_30d": 550.0, "adv_usd_30d": 15.0, "notional_usd_30d": 550.0, "bars": 100, "active_days": 30},
    ]

    monkeypatch.setattr("training.universe.top50._rank_symbols_by_liquidity", lambda **_: ranked)

    prev = tmp_path / "prev_snapshot.json"
    prev.write_text(json.dumps({"symbols": ["A", "B", "C"]}), encoding="utf-8")
    cfg = UniverseBuildConfig(
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
        top_n=3,
        lookback_days=30,
        timeframe="1h",
        min_notional_usd=1.0,
        exclude_stable=True,
        exclude_leveraged=True,
        hysteresis_keep_rank=4,
        track="liquid",
        rank_by="volume_usd_30d",
        source="db",
    )
    snap = build_top_universe_snapshot(database_url="postgresql://unused", cfg=cfg, previous_snapshot_file=prev)
    assert snap["symbols"] == ["A", "B", "D"]
    rows = {str(r["symbol"]): r for r in list(snap["symbol_rows"])}
    assert bool(rows["B"]["kept_by_hysteresis"]) is True
    assert bool(rows["A"]["kept_by_hysteresis"]) is False

