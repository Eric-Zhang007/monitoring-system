from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402


def test_normalize_reject_reason_categories():
    assert router_mod._normalize_reject_reason("invalid_quantity") == "invalid_quantity"
    assert router_mod._normalize_reject_reason("venue_error:Timeout") == "venue_error"
    assert router_mod._normalize_reject_reason("paper_reject_simulated") == "simulated_reject"
    assert router_mod._normalize_reject_reason("") == "none"


def test_normalize_execution_payload_sets_category():
    out = router_mod._normalize_execution_payload({"status": "rejected", "reject_reason": "slippage_too_wide", "lifecycle": []})
    assert out["reject_reason_category"] == "slippage_too_wide"


def test_replay_backtest_emits_daily_sharpe_fields():
    now = datetime.now(timezone.utc) - timedelta(days=3)
    feature_rows = []
    price_rows = []
    price = 100.0
    for i in range(120):
        ts = now + timedelta(hours=i)
        price = price * (1.0 + (0.002 if i % 3 == 0 else -0.001))
        price_rows.append({"price": price, "volume": 2000.0 + i, "timestamp": ts})
        if i < 119:
            feature_rows.append(
                {
                    "lineage_id": f"ln-{i}",
                    "as_of_ts": ts,
                    "feature_payload": {
                        "ret_1": 0.001,
                        "ret_3": 0.002,
                        "ret_12": 0.003,
                        "ret_48": 0.001,
                        "vol_3": 0.01,
                        "vol_12": 0.01,
                        "vol_48": 0.01,
                        "vol_96": 0.01,
                        "log_volume": 7.0,
                        "vol_z": 0.1,
                        "volume_impact": 0.001,
                        "orderbook_imbalance": 0.0,
                        "funding_rate": 0.0,
                        "onchain_norm": 0.0,
                        "event_decay": 0.4,
                    },
                }
            )
    out = router_mod._run_model_replay_backtest(
        feature_rows=feature_rows,
        price_rows=price_rows,
        fee_bps=5.0,
        slippage_bps=3.0,
    )
    assert out["status"] == "completed"
    assert "sharpe_step_raw" in out
    assert "sharpe_daily" in out
    assert out["sharpe_method"] == "daily_agg_v1"
    assert int(out["observation_days"]) >= 2
