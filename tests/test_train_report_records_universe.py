from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import training.train_liquid as train_liquid


def test_train_report_records_universe(monkeypatch):
    class _FakeRepo:
        def __init__(self, db_url: str):
            self.db_url = db_url
            self.upsert_called = False

        def resolve_asset_universe_asof(self, track, as_of, fallback_targets):
            _ = (as_of, fallback_targets)
            return {
                "track": track,
                "as_of": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
                "symbols": ["BTC", "ETH", "SOL"],
                "source": "snapshot",
                "universe_version": "liquid-v20260224",
                "snapshot_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
            }

        def upsert_asset_universe_snapshot(self, **kwargs):
            self.upsert_called = True
            return kwargs

    monkeypatch.setattr(train_liquid, "V2Repository", _FakeRepo)
    cfg = train_liquid.TrainConfig(
        db_url="postgresql://unused",
        symbols=["btc", "eth"],
        universe_track="liquid",
        use_universe_snapshot=True,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 2, tzinfo=timezone.utc),
        lookback=16,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-4,
        d_model=64,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
        patch_len=4,
        backbone="patchtst",
        max_samples_per_symbol=0,
        out_dir=Path("artifacts/models/liquid_main"),
        model_id="liquid_main",
        cost_profile="standard",
        train_days=10,
        val_days=2,
        test_days=2,
        purge_gap_hours=1,
        step_days=2,
        force_purged=True,
    )
    universe = train_liquid._resolve_training_universe(cfg)
    report = {"status": "ok"}
    train_liquid._attach_universe_to_training_report(report, universe)
    assert report["universe"]["universe_version"] == "liquid-v20260224"
    assert report["universe"]["source"] == "snapshot"
    assert report["universe"]["symbols"] == ["BTC", "ETH", "SOL"]

