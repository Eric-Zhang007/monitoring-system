from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from execution_engine import PaperExecutionAdapter  # noqa: E402


def test_paper_execution_reject_rate_under_one_percent():
    adapter = PaperExecutionAdapter()
    total = 1000
    rejected = 0
    reasons = set()
    order = {"target": "BTC", "side": "buy", "quantity": 0.01, "est_price": 50000.0}
    context = {"max_slippage_bps": 20.0, "limit_timeout_sec": 2.0, "max_retries": 1, "fee_bps": 5.0}
    for _ in range(total):
        out = adapter.execute(order, context=context)
        if str(out.get("status")) == "rejected":
            rejected += 1
            reasons.add(str(out.get("reject_reason") or ""))
    assert rejected / total < 0.01
    assert "paper_reject_simulated" not in reasons
