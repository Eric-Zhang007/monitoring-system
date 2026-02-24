from __future__ import annotations

from pathlib import Path


def test_init_db_contains_asset_universe_snapshots_table():
    sql = Path("scripts/init_db.sql").read_text(encoding="utf-8").lower()
    assert "create table if not exists asset_universe_snapshots" in sql
    assert "idx_asset_universe_snapshots_track_asof" in sql

