from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "monitoring"))

hc = pytest.importorskip("health_check")


class _FakeCursor:
    def __init__(self):
        self.last_query = ""

    def execute(self, query, params=None):
        self.last_query = str(query)

    def fetchone(self):
        if "SELECT version()" in self.last_query:
            return ("PostgreSQL 16",)
        return (0,)


class _FakeConn:
    def __init__(self):
        self._cursor = _FakeCursor()
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def test_check_postgres_graceful_when_optional_tables_missing(monkeypatch):
    fake_conn = _FakeConn()
    monkeypatch.setattr(hc.psycopg2, "connect", lambda _dsn: fake_conn)

    def _fake_safe_recent_count(cursor, *, table_fqn, time_columns, lookback_hours):
        return None, f"optional probe skipped: {table_fqn} table missing"

    monkeypatch.setattr(hc, "_safe_recent_count", _fake_safe_recent_count)

    healthy, data = asyncio.run(hc.check_postgres())
    assert healthy is True
    assert data["recent_predictions"] is None
    assert data["recent_features"] is None
    assert len(data.get("warnings") or []) == 2
    assert fake_conn.closed is True


def test_safe_recent_count_handles_missing_time_column(monkeypatch):
    cursor = _FakeCursor()
    monkeypatch.setattr(hc, "_table_exists", lambda _cursor, _table: True)
    monkeypatch.setattr(hc, "_table_columns", lambda _cursor, _schema, _table: {"id", "value"})
    count, warning = hc._safe_recent_count(
        cursor,
        table_fqn="public.semantic_features",
        time_columns=("timestamp", "created_at"),
        lookback_hours=6,
    )
    assert count is None
    assert "missing time column" in str(warning)


def test_safe_recent_count_returns_value_when_available(monkeypatch):
    cursor = _FakeCursor()
    monkeypatch.setattr(hc, "_table_exists", lambda _cursor, _table: True)
    monkeypatch.setattr(hc, "_table_columns", lambda _cursor, _schema, _table: {"created_at"})
    count, warning = hc._safe_recent_count(
        cursor,
        table_fqn="public.predictions_v2",
        time_columns=("created_at", "timestamp"),
        lookback_hours=1,
    )
    assert count == 0
    assert warning is None


def test_safe_recent_count_graceful_when_table_lookup_fails(monkeypatch):
    cursor = _FakeCursor()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(hc, "_table_exists", _boom)
    count, warning = hc._safe_recent_count(
        cursor,
        table_fqn="public.predictions_v2",
        time_columns=("created_at", "timestamp"),
        lookback_hours=1,
    )
    assert count is None
    assert "lookup failed" in str(warning)


def test_safe_recent_count_graceful_when_columns_lookup_fails(monkeypatch):
    cursor = _FakeCursor()
    monkeypatch.setattr(hc, "_table_exists", lambda _cursor, _table: True)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("columns metadata unavailable")

    monkeypatch.setattr(hc, "_table_columns", _boom)
    count, warning = hc._safe_recent_count(
        cursor,
        table_fqn="public.semantic_features",
        time_columns=("timestamp", "created_at"),
        lookback_hours=6,
    )
    assert count is None
    assert "columns lookup failed" in str(warning)
