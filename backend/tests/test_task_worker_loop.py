from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import types

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "monitoring"))

tw = pytest.importorskip("task_worker")


def test_run_backtest_job_runs_inside_active_event_loop(monkeypatch):
    class _FakeResp:
        def model_dump(self):
            return {"status": "ok"}

    async def _fake_run_backtest(req):
        loop = asyncio.get_running_loop()
        assert loop.is_running()
        assert req.track == "liquid"
        return _FakeResp()

    fake_module = types.SimpleNamespace(run_backtest=_fake_run_backtest)
    monkeypatch.setitem(sys.modules, "v2_router", fake_module)

    async def _case():
        out = await tw._run_backtest_job({"track": "liquid"})
        assert out == {"status": "ok"}

    asyncio.run(_case())


def test_run_worker_marks_completed(monkeypatch):
    tasks = [
        {
            "task_id": "task-1",
            "task_type": "backtest_run",
            "payload": {"track": "liquid"},
        }
    ]
    status_events = []

    def _fake_claim_next_task(timeout_seconds=5):
        return tasks.pop(0) if tasks else None

    def _fake_set_task_status(task_id, status, *, result=None, error=None):
        status_events.append({"task_id": task_id, "status": status, "result": result, "error": error})

    def _fake_execute_task(task_type, payload, *, backtest_runtime=None):
        assert task_type == "backtest_run"
        assert payload["track"] == "liquid"
        return {"ok": True}

    monkeypatch.setattr(tw, "claim_next_task", _fake_claim_next_task)
    monkeypatch.setattr(tw, "set_task_status", _fake_set_task_status)
    monkeypatch.setattr(tw, "_execute_task", _fake_execute_task)

    tw.run_worker(max_tasks=1)
    assert [evt["status"] for evt in status_events] == ["running", "completed"]
    assert status_events[-1]["result"] == {"ok": True}


def test_run_worker_marks_failed_on_exception(monkeypatch):
    tasks = [
        {
            "task_id": "task-2",
            "task_type": "backtest_run",
            "payload": {"track": "liquid"},
        }
    ]
    status_events = []

    def _fake_claim_next_task(timeout_seconds=5):
        return tasks.pop(0) if tasks else None

    def _fake_set_task_status(task_id, status, *, result=None, error=None):
        status_events.append({"task_id": task_id, "status": status, "result": result, "error": error})

    def _fake_execute_task(task_type, payload, *, backtest_runtime=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(tw, "claim_next_task", _fake_claim_next_task)
    monkeypatch.setattr(tw, "set_task_status", _fake_set_task_status)
    monkeypatch.setattr(tw, "_execute_task", _fake_execute_task)

    tw.run_worker(max_tasks=1)
    assert [evt["status"] for evt in status_events] == ["running", "failed"]
    assert "boom" in str(status_events[-1]["error"])


def test_task_worker_class_uses_injected_io(monkeypatch):
    tasks = [{"task_id": "task-3", "task_type": "pnl_attribution", "payload": {"track": "liquid"}}]
    status_events = []

    def _fake_claim_next_task(timeout_seconds=5):
        assert timeout_seconds == 9
        return tasks.pop(0) if tasks else None

    def _fake_set_task_status(task_id, status, *, result=None, error=None):
        status_events.append({"task_id": task_id, "status": status, "result": result, "error": error})

    def _fake_execute_task(task_type, payload, *, backtest_runtime=None):
        assert task_type == "pnl_attribution"
        assert payload["track"] == "liquid"
        return {"ok": True}

    monkeypatch.setattr(tw, "_execute_task", _fake_execute_task)

    worker = tw.TaskWorker(
        poll_timeout_sec=9,
        claim_task_fn=_fake_claim_next_task,
        set_status_fn=_fake_set_task_status,
    )
    worker.run(max_tasks=1)
    assert [evt["status"] for evt in status_events] == ["running", "completed"]
    assert status_events[-1]["result"] == {"ok": True}
