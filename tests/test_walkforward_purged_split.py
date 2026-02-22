from __future__ import annotations

from datetime import datetime, timedelta, timezone

from training.splits.walkforward_purged import WalkForwardPurgedConfig, build_walkforward_purged_splits


def test_walkforward_purged_gap_applied():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = [start + timedelta(hours=1) * i for i in range(24 * 40)]
    cfg = WalkForwardPurgedConfig(train_days=10, val_days=3, test_days=3, purge_gap_hours=24, step_days=3)
    folds = build_walkforward_purged_splits(ts, cfg)
    assert folds
    f0 = folds[0]
    assert f0["train_end"] <= f0["val_start"] - timedelta(hours=24)
    assert f0["val_end"] <= f0["test_start"] - timedelta(hours=24)
