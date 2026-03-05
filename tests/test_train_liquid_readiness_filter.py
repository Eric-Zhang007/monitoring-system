from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import training.train_liquid as train_liquid


def test_audit_blocked_symbols_are_excluded(monkeypatch):
    cfg = train_liquid.TrainConfig(
        db_url="postgresql://unused",
        symbols=["BTC", "ETH", "SOL"],
        universe_track="liquid",
        use_universe_snapshot=False,
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
        top_n=50,
        enable_offline_audit=True,
        audit_enforce_blocked_exclusion=True,
        audit_bucket="5m",
    )

    def _fake_run_audit(_args):  # noqa: ANN001
        return {
            "task_id": "audit-1",
            "summary": {"READY": 1, "DEGRADED": 1, "BLOCKED": 1},
            "ready_symbols": ["BTC"],
            "degraded_symbols": ["ETH"],
            "blocked_symbols": ["SOL"],
        }

    monkeypatch.setattr(train_liquid, "run_audit", _fake_run_audit)
    out = train_liquid._audit_and_filter_symbols(cfg, ["BTC", "ETH", "SOL"])
    assert out["symbols_kept"] == ["BTC", "ETH"]
    assert out["symbols_blocked"] == ["SOL"]
    assert out["audit_summary"]["BLOCKED"] == 1

