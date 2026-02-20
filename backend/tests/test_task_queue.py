from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import task_queue as tq  # noqa: E402


class _FakeRedis:
    def __init__(self):
        self.kv = {}
        self.lists = {}

    def setex(self, key, ttl, val):
        self.kv[key] = val

    def lpush(self, key, val):
        self.lists.setdefault(key, []).insert(0, val)

    def brpop(self, key, timeout=1):
        q = self.lists.setdefault(key, [])
        if not q:
            return None
        val = q.pop()
        return key, val

    def brpoplpush(self, source, destination, timeout=1):
        src = self.lists.setdefault(source, [])
        if not src:
            return None
        val = src.pop()
        self.lists.setdefault(destination, []).insert(0, val)
        return val

    def lrange(self, key, start, end):
        values = list(self.lists.setdefault(key, []))
        if end == -1:
            end = len(values) - 1
        if start < 0:
            start = max(0, len(values) + start)
        if end < 0:
            end = len(values) + end
        if start > end:
            return []
        return values[start : end + 1]

    def lrem(self, key, count, value):
        values = self.lists.setdefault(key, [])
        removed = 0
        if count == 0:
            new_values = [v for v in values if not (v == value and (removed := removed + 1))]
            self.lists[key] = new_values
            return removed
        if count > 0:
            new_values = []
            for v in values:
                if v == value and removed < count:
                    removed += 1
                    continue
                new_values.append(v)
            self.lists[key] = new_values
            return removed
        # count < 0: remove from tail
        to_remove = abs(count)
        values_rev = list(reversed(values))
        new_rev = []
        for v in values_rev:
            if v == value and removed < to_remove:
                removed += 1
                continue
            new_rev.append(v)
        self.lists[key] = list(reversed(new_rev))
        return removed

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


def test_claim_moves_task_to_processing_until_terminal_status(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(tq, "_client", lambda: fake)

    task_id = tq.enqueue_task("backtest_run", {"track": "liquid"})
    claimed = tq.claim_next_task(timeout_seconds=1)

    assert claimed is not None
    assert claimed["task_id"] == task_id
    assert len(fake.lrange(tq.TASK_PROCESSING_KEY, 0, -1)) == 1

    tq.set_task_status(task_id, "completed", result={"ok": True})
    assert len(fake.lrange(tq.TASK_PROCESSING_KEY, 0, -1)) == 0


def test_stale_processing_task_is_requeued(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(tq, "_client", lambda: fake)
    monkeypatch.setattr(tq, "TASK_PROCESSING_TIMEOUT_SECONDS", 1)

    task_id = tq.enqueue_task("pnl_attribution", {"track": "liquid"})
    first_claim = tq.claim_next_task(timeout_seconds=1)
    assert first_claim is not None
    assert first_claim["task_id"] == task_id

    raw = fake.lrange(tq.TASK_PROCESSING_KEY, 0, -1)[0]
    obj = json.loads(raw)
    obj["claimed_at"] = "2000-01-01T00:00:00+00:00"
    fake.lists[tq.TASK_PROCESSING_KEY][0] = json.dumps(obj)

    second_claim = tq.claim_next_task(timeout_seconds=1)
    assert second_claim is not None
    assert second_claim["task_id"] == task_id
    assert int(second_claim.get("requeue_count") or 0) >= 1
