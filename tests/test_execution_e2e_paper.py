from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import sys
import types
from typing import Any, Dict, List

# v2_router imports model modules that depend on torch; this test only exercises
# execution endpoints and can run with a tiny torch stub.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    nn_stub = types.ModuleType("torch.nn")
    nn_func_stub = types.ModuleType("torch.nn.functional")

    class _Module:
        pass

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    nn_stub.Module = _Module
    torch_stub.nn = nn_stub
    torch_stub.Tensor = object
    torch_stub.float32 = "float32"
    torch_stub.no_grad = lambda: _NoGrad()
    torch_stub.tensor = lambda *args, **kwargs: None
    torch_stub.load = lambda *args, **kwargs: {}
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.nn.functional"] = nn_func_stub

import v2_router as router_mod
from execution_engine import ExecutionEngine
from schemas_v2 import ExecuteOrdersRequest, ExecutionOrderInput, SubmitExecutionOrdersRequest


class _FakeRepo:
    def __init__(self):
        self._next_order_id = 1
        self._next_child_id = 1
        self._next_fill_id = 1
        self.orders: Dict[int, Dict[str, Any]] = {}
        self.decisions: Dict[str, Dict[str, Any]] = {}
        self.child_orders: Dict[int, Dict[str, Any]] = {}
        self.fills: List[Dict[str, Any]] = []
        self.positions_live: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        now = datetime.now(timezone.utc)
        self.orderbooks = {
            "BTC": {"symbol": "BTC", "ts": now, "bid_px": 9999.0, "ask_px": 10001.0, "spread_bps": 2.0, "imbalance": 0.05},
            "ETH": {"symbol": "ETH", "ts": now, "bid_px": 1999.0, "ask_px": 2001.0, "spread_bps": 10.0, "imbalance": -0.02},
        }

    def create_execution_decision(self, payload: Dict[str, Any]) -> str:
        d = dict(payload)
        decision_id = str(d["decision_id"])
        self.decisions[decision_id] = {
            **d,
            "decision_id": decision_id,
            "status": str(d.get("status") or "created"),
            "trace_summary": dict(d.get("trace_summary") or {}),
        }
        return decision_id

    def get_execution_decision(self, decision_id: str):
        return dict(self.decisions.get(decision_id) or {})

    def start_execution_decision(self, decision_id: str) -> None:
        self.decisions[decision_id]["status"] = "running"
        self.decisions[decision_id]["started_at"] = datetime.now(timezone.utc)

    def finish_execution_decision(self, decision_id: str, status: str, summary: Dict[str, Any], error: str | None = None) -> None:
        self.decisions[decision_id]["status"] = status
        self.decisions[decision_id]["trace_summary"] = dict(summary)
        self.decisions[decision_id]["error"] = error
        self.decisions[decision_id]["ended_at"] = datetime.now(timezone.utc)

    def create_execution_orders(self, decision_id: str, adapter: str, venue: str, time_in_force: str, max_slippage_bps: float, orders: List[Dict[str, Any]]):
        out = []
        for o in orders:
            oid = self._next_order_id
            self._next_order_id += 1
            self.orders[oid] = {
                "id": oid,
                "decision_id": decision_id,
                "target": str(o["target"]).upper(),
                "track": str(o.get("track") or "liquid"),
                "side": str(o["side"]),
                "quantity": float(o.get("quantity") or 0.0),
                "est_price": float(o.get("est_price") or 0.0),
                "status": "submitted",
                "adapter": adapter,
                "venue": venue,
                "time_in_force": time_in_force,
                "max_slippage_bps": float(max_slippage_bps),
                "strategy_id": str(o.get("strategy_id") or "default"),
                "metadata": dict(o.get("metadata") or {}),
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "fees_paid": 0.0,
                "reject_reason": None,
                "created_at": datetime.now(timezone.utc),
            }
            out.append(oid)
        return out

    def fetch_orders_for_decision(self, decision_id: str, limit: int = 100):
        rows = [dict(v) for v in self.orders.values() if str(v.get("decision_id")) == str(decision_id)]
        rows.sort(key=lambda x: int(x["id"]))
        return rows[:limit]

    def get_model_rollout_state(self, track: str):
        _ = track
        return {"stage_pct": 100}

    def is_kill_switch_triggered(self, track: str, strategy_id: str) -> bool:
        _ = (track, strategy_id)
        return False

    def load_price_history(self, symbol: str, lookback_days: int = 2):
        _ = lookback_days
        now = datetime.now(timezone.utc)
        base = 10000.0 if symbol.upper() == "BTC" else 2000.0
        return [{"price": base * (1.0 + i * 0.00005), "timestamp": now - timedelta(minutes=120 - i)} for i in range(120)]

    def latest_price_snapshot(self, symbol: str):
        sym = str(symbol).upper()
        if sym in {"BTC", "ETH"}:
            p = 10000.0 if sym == "BTC" else 2000.0
            return {"symbol": sym, "price": p, "volume": 100000.0, "timestamp": datetime.now(timezone.utc)}
        return None

    def get_execution_consecutive_losses(self, track: str, lookback_hours: int = 24, limit: int = 200, strategy_id: str | None = None):
        _ = (track, lookback_hours, limit, strategy_id)
        return 0

    def get_execution_edge_pnls(self, track: str, lookback_hours: int, limit: int = 500, strategy_id: str | None = None):
        _ = (track, lookback_hours, limit, strategy_id)
        return []

    def build_pnl_attribution(self, track: str, lookback_hours: int):
        _ = (track, lookback_hours)
        return {"totals": {"net_pnl": 0.0, "gross_notional_signed": 0.0}}

    def save_risk_event(self, **kwargs):
        _ = kwargs
        return None

    def upsert_kill_switch_state(self, **kwargs):
        _ = kwargs
        return {}

    def latest_orderbook_l2(self, symbol: str):
        return dict(self.orderbooks.get(str(symbol).upper()) or {})

    def create_child_orders(self, parent_order_id: int, child_rows: List[Dict[str, Any]]):
        out = []
        for row in child_rows:
            cid = self._next_child_id
            self._next_child_id += 1
            self.child_orders[cid] = {
                "id": cid,
                "parent_order_id": int(parent_order_id),
                **dict(row),
                "status": str(row.get("status") or "new"),
                "lifecycle": list(row.get("lifecycle") or []),
            }
            out.append(cid)
        return out

    def update_child_order_status(self, child_id: int, status: str, venue_order_id: str | None = None, lifecycle_append: Dict[str, Any] | None = None):
        row = self.child_orders[int(child_id)]
        row["status"] = str(status)
        if venue_order_id:
            row["venue_order_id"] = venue_order_id
        if lifecycle_append:
            row["lifecycle"] = list(row.get("lifecycle") or []) + [dict(lifecycle_append)]
        self.child_orders[int(child_id)] = row

    def insert_execution_fills(self, child_id: int, fills: List[Dict[str, Any]]):
        ids: List[int] = []
        for fill in fills:
            fid = self._next_fill_id
            self._next_fill_id += 1
            row = dict(fill)
            row["id"] = fid
            row["child_order_id"] = int(child_id)
            ids.append(fid)
            self.fills.append(row)
        return ids

    def update_parent_from_fills(self, parent_order_id: int):
        parent = self.orders[int(parent_order_id)]
        child_ids = [cid for cid, c in self.child_orders.items() if int(c.get("parent_order_id")) == int(parent_order_id)]
        rel = [f for f in self.fills if int(f.get("child_order_id")) in child_ids]
        filled_qty = sum(float(f.get("qty") or 0.0) for f in rel)
        notional = sum(float(f.get("qty") or 0.0) * float(f.get("price") or 0.0) for f in rel)
        fees = sum(float(f.get("fee") or 0.0) for f in rel)
        avg = notional / filled_qty if filled_qty > 1e-12 else None
        if filled_qty >= float(parent.get("quantity") or 0.0) - 1e-12:
            status = "filled"
        elif filled_qty > 1e-12:
            status = "partially_filled"
        else:
            status = "submitted"
        parent.update({"status": status, "filled_qty": filled_qty, "avg_fill_price": avg, "fees_paid": fees})
        self.orders[int(parent_order_id)] = parent
        return {"status": status, "filled_qty": filled_qty, "avg_fill_price": avg, "fees_paid": fees, "last_venue_order_id": None}

    def update_order_execution(self, order_id: int, status: str, metadata: Dict[str, Any]):
        row = self.orders[int(order_id)]
        row["status"] = str(status)
        row["metadata"] = {**dict(row.get("metadata") or {}), **dict(metadata or {})}
        execution = metadata.get("execution") if isinstance(metadata.get("execution"), dict) else {}
        row["filled_qty"] = float(execution.get("filled_qty") or row.get("filled_qty") or 0.0)
        row["avg_fill_price"] = execution.get("avg_fill_price", row.get("avg_fill_price"))
        row["fees_paid"] = float(execution.get("fees_paid") or row.get("fees_paid") or 0.0)
        row["reject_reason"] = execution.get("reject_reason")
        self.orders[int(order_id)] = row

    def get_position_live(self, venue: str, symbol: str, account_id: str = ""):
        return self.positions_live.get((str(venue), str(symbol).upper(), str(account_id)))

    def upsert_position_live(self, *, venue: str, symbol: str, account_id: str = "", position_qty: float, avg_cost: float, unrealized_pnl: float, raw: Dict[str, Any] | None = None):
        self.positions_live[(str(venue), str(symbol).upper(), str(account_id))] = {
            "venue": str(venue),
            "symbol": str(symbol).upper(),
            "account_id": str(account_id),
            "position_qty": float(position_qty),
            "avg_cost": float(avg_cost),
            "unrealized_pnl": float(unrealized_pnl),
            "raw": dict(raw or {}),
        }


def test_execution_e2e_paper_submit_run_trace_and_position(monkeypatch):
    fake_repo = _FakeRepo()
    monkeypatch.setattr(router_mod, "repo", fake_repo)
    monkeypatch.setattr(router_mod, "exec_engine", ExecutionEngine())
    monkeypatch.setenv("PAPER_SEED", "42")
    submit = SubmitExecutionOrdersRequest(
        adapter="paper",
        venue="coinbase",
        time_in_force="IOC",
        max_slippage_bps=15.0,
        market_type="spot",
        orders=[
            ExecutionOrderInput(target="BTC", side="buy", quantity=0.20, est_price=10000.0, strategy_id="s1", metadata={"horizon": "1h"}),
            ExecutionOrderInput(target="ETH", side="sell", quantity=0.50, est_price=2000.0, strategy_id="s1", metadata={"horizon": "1d"}),
        ],
    )
    submit_resp = asyncio.run(router_mod.submit_execution_orders(submit))
    assert submit_resp.accepted_orders == 2

    run_req = ExecuteOrdersRequest(
        decision_id=submit_resp.decision_id,
        adapter="paper",
        time_in_force="IOC",
        max_slippage_bps=15.0,
        venue="coinbase",
        market_type="spot",
        max_orders=10,
        limit_timeout_sec=1.2,
        max_retries=1,
        fee_bps=5.0,
        risk_equity_usd=10000.0,
    )
    run_resp = asyncio.run(router_mod.run_execution(run_req))
    assert run_resp.total == 2
    assert run_resp.filled + run_resp.rejected == 2
    assert len(run_resp.orders) == 2

    assert fake_repo.child_orders
    assert fake_repo.fills
    for row in run_resp.orders:
        assert isinstance(row.get("execution_trace"), dict)
        assert isinstance(row.get("child_orders"), list)
        assert isinstance(row.get("fills"), list)
        trace = row.get("execution_trace") or {}
        if int(trace.get("market_state_missing") or 0) == 0:
            assert abs(float(trace.get("slippage_bps") or 0.0)) <= 15.0 + 1e-6

    summary = fake_repo.decisions[submit_resp.decision_id].get("trace_summary") or {}
    assert int(summary.get("total") or 0) == run_resp.total
    assert int(summary.get("filled") or 0) == run_resp.filled
    assert int(summary.get("rejected") or 0) == run_resp.rejected

    assert ("coinbase", "BTC", "") in fake_repo.positions_live
    assert ("coinbase", "ETH", "") in fake_repo.positions_live
