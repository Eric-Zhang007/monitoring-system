from __future__ import annotations

from monitoring.reconciliation_daemon import run_once


class _Repo:
    def __init__(self):
        self.kill_switch = []
        self.logs = []
        self.risk_events = []

    def list_open_child_orders_live(self, max_age_sec=None):
        _ = max_age_sec
        return [{"id": 1, "adapter": "paper", "venue": "coinbase", "venue_order_id": "oid-1", "parent_order_id": 11}]

    def update_child_order_status(self, *args, **kwargs):
        _ = (args, kwargs)
        return None

    def insert_execution_fills(self, *args, **kwargs):
        _ = (args, kwargs)
        return []

    def update_parent_from_fills(self, *args, **kwargs):
        _ = (args, kwargs)
        return {"status": "submitted", "filled_qty": 0.0}

    def latest_price_snapshot(self, symbol):
        return {"symbol": symbol, "price": 100.0}

    def list_positions_live(self, venue):
        _ = venue
        return []

    def append_reconciliation_log(self, **kwargs):
        self.logs.append(dict(kwargs))
        return 1

    def upsert_kill_switch_state(self, **kwargs):
        self.kill_switch.append(dict(kwargs))
        return {}

    def save_risk_event(self, **kwargs):
        self.risk_events.append(dict(kwargs))
        return None


class _FailAdapter:
    def poll_order(self, venue_order_id, timeout):
        _ = (venue_order_id, timeout)
        raise RuntimeError("poll_failed")

    def fetch_fills(self, venue_order_id):
        _ = venue_order_id
        return []

    def fetch_positions(self):
        return []


class _Engine:
    def __init__(self):
        self.adapters = {"paper": _FailAdapter()}


def test_recon_failure_triggers_kill_switch():
    repo = _Repo()
    out = run_once(
        repo=repo,
        engine=_Engine(),
        max_age_sec=60,
        pos_diff_usd_max=9999.0,
        fail_triggers_kill=True,
        fails_to_red=1,
    )
    assert int(out["poll_failures"]) >= 1
    assert any(str(x.get("reason")) == "reconciliation_poll_failures" for x in repo.kill_switch)

