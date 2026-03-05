from __future__ import annotations

import sys
import time
from typing import Any, Dict, List

from backend.process_manager import ProcessManager


class _FakeRepo:
    def __init__(self):
        self.process_rows: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []

    def upsert_ops_process(self, process_id: str, payload: Dict[str, Any]):
        row = dict(payload)
        row["process_id"] = process_id
        self.process_rows[process_id] = row
        return row

    def append_ops_process_event(self, process_id: str, event_type: str, payload: Dict[str, Any]):
        self.events.append({"process_id": process_id, "event_type": event_type, "payload": dict(payload)})
        return len(self.events)


def test_process_manager_start_and_stop():
    repo = _FakeRepo()
    pm = ProcessManager(repo)  # type: ignore[arg-type]
    try:
        started = pm.start(
            task_type="TRAIN_LIQUID",
            params={
                "command": [sys.executable, "-c", "import time; time.sleep(10)"],
                "track": "liquid",
                "symbols": ["BTC", "ETH"],
            },
            created_by="pytest",
        )
        assert started["status"] == "running"
        assert int(started["pid"] or 0) > 0
        pid = str(started["process_id"])
        listed = pm.list_processes()
        assert any(str(x.get("process_id")) == pid for x in listed)

        stopped = pm.stop(pid, force=False)
        assert stopped["status"] == "stopped"
        assert len(repo.events) >= 2
    finally:
        # ensure background thread exits even if assert fails
        pm.shutdown()
