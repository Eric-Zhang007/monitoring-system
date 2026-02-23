from __future__ import annotations

from datetime import timedelta

import pytest

from account_state.aggregator import AccountStateAggregator


class _Adapter:
    name = "paper"

    def fetch_balances(self):
        return {
            "cash": 1000.0,
            "equity": 1000.0,
            "free_margin": 900.0,
            "used_margin": 100.0,
            "margin_ratio": 10.0,
            "account_currency": "USD",
        }

    def fetch_positions(self):
        return []

    def fetch_open_orders(self):
        return []


def test_account_state_ttl_enforced():
    agg = AccountStateAggregator(adapter=_Adapter(), venue="coinbase", store=None, cache_ttl_s=1)
    state = agg.refresh_full_state()
    state.ts = state.ts - timedelta(seconds=5)
    with pytest.raises(RuntimeError, match="account_state_stale"):
        agg.get_state(require_fresh=True)

