from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import task_queue as tq  # noqa: E402


class _FakeRedis:
    def __init__(self):
        self.kv = {}
        self.q = []

    def setex(self, key, ttl, val):
        self.kv[key] = val

    def lpush(self, key, val):
        self.q.insert(0, val)

    def brpop(self, key, timeout=1):
        if not self.q:
            return None
        val = self.q.pop()
        return key, val

    def get(self, key):
        return self.kv.get(key)


def test_enqueue_and_get_task(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(tq, "_client", lambda: fake)

    task_id = tq.enqueue_task("backtest_run", {"track": "liquid"})
    data = tq.get_task(task_id)

    assert task_id.startswith("task-")
    assert data is not None
    assert data["task_type"] == "backtest_run"
    assert data["status"] == "queued"


def test_claim_and_update(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(tq, "_client", lambda: fake)

    task_id = tq.enqueue_task("pnl_attribution", {"track": "liquid"})
    claimed = tq.claim_next_task(timeout_seconds=1)
    assert claimed is not None
    assert claimed["task_id"] == task_id

    tq.set_task_status(task_id, "running")
    tq.set_task_status(task_id, "completed", result={"ok": True})

    data = tq.get_task(task_id)
    assert data is not None
    assert data["status"] == "completed"
    assert data["result"]["ok"] is True
