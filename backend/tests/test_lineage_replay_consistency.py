from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from v2_router import _run_model_replay_backtest  # noqa: E402


def _build_feature_rows(n: int):
    out = []
    for i in range(n):
        out.append(
            {
                "lineage_id": "ln-a" if i % 2 == 0 else "ln-b",
                "feature_payload": {
                    "ret_1": 0.001 if i % 3 == 0 else -0.0005,
                    "ret_3": 0.002,
                    "ret_12": 0.004,
                    "ret_48": -0.001,
                    "orderbook_imbalance": 0.1,
                    "funding_rate": 0.0001,
                    "onchain_norm": 0.02,
                    "event_decay": 0.03,
                    "vol_12": 0.02,
                },
            }
        )
    return out


def _build_price_rows(n: int):
    price = 100.0
    rows = []
    for i in range(n):
        price = price * (1.0 + (0.001 if i % 2 == 0 else -0.0008))
        rows.append({"price": price, "volume": 1000.0 + i})
    return rows


def test_model_replay_backtest_is_repeatable():
    features = _build_feature_rows(120)
    prices = _build_price_rows(121)
    out1 = _run_model_replay_backtest(features, prices, fee_bps=5.0, slippage_bps=3.0)
    out2 = _run_model_replay_backtest(features, prices, fee_bps=5.0, slippage_bps=3.0)
    assert out1["status"] == "completed"
    assert out2["status"] == "completed"
    assert out1["pnl_after_cost"] == out2["pnl_after_cost"]
    assert out1["lineage_coverage"] == out2["lineage_coverage"]


def test_model_replay_backtest_reports_lineage_and_cost_breakdown():
    features = _build_feature_rows(80)
    prices = _build_price_rows(81)
    out = _run_model_replay_backtest(features, prices, fee_bps=5.0, slippage_bps=3.0)
    assert "lineage_coverage" in out
    assert "cost_breakdown" in out
    assert set(out["cost_breakdown"].keys()) == {"fee", "slippage", "impact"}
