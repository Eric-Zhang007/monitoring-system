from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from v2_repository import V2Repository  # noqa: E402


class _FakeCursor:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):  # noqa: ANN001
        self.calls.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _FakeConn:
    def __init__(self, cur: _FakeCursor):
        self.cur = cur

    def cursor(self):
        return self.cur

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


def test_mark_backtest_run_superseded_executes_update(monkeypatch):
    repo = V2Repository.__new__(V2Repository)
    cur = _FakeCursor()

    def _fake_connect():
        return _FakeConn(cur)

    monkeypatch.setattr(repo, "_connect", _fake_connect)
    repo.mark_backtest_run_superseded(run_id=10, superseded_by_run_id=20, reason="artifact_replayed_completed")

    assert cur.calls
    sql, params = cur.calls[0]
    assert "UPDATE backtest_runs" in sql
    assert params == (20, "artifact_replayed_completed", 10)
