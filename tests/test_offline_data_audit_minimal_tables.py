from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import scripts.audit_offline_training_data as audit_mod


class _FakeCursor:
    def __init__(self):
        self._rows: List[Dict[str, Any]] = []
        self._one: Dict[str, Any] = {}
        self.exists = {
            "market_bars": True,
            "feature_snapshots_main": True,
            "feature_matrix_main": True,
            "asset_universe_snapshots": True,
            "social_posts_raw": False,
            "social_comments_raw": False,
            "events": False,
            "social_text_embeddings": False,
            "offline_data_audits": True,
        }
        self.now = datetime.now(timezone.utc)

    def execute(self, sql: str, params=None):  # noqa: ANN001
        q = " ".join(str(sql).lower().split())
        p = params or ()
        self._rows = []
        self._one = {}
        if "select to_regclass" in q:
            table = str(p[0]).split(".")[-1]
            self._one = {"reg": f"public.{table}" if self.exists.get(table, False) else None}
            return
        if "select column_name from information_schema.columns" in q:
            table = str(p[0])
            if table in {"social_posts_raw", "social_comments_raw", "events"}:
                self._rows = [{"column_name": "created_at"}]
            else:
                self._rows = []
            return
        if "select distinct floor(extract(epoch from ts)" in q and "from market_bars" in q:
            self._rows = [{"b": int((self.now - timedelta(minutes=5 * i)).timestamp() // 300)} for i in range(20)]
            return
        if "select min(ts) as min_ts, max(ts) as max_ts from market_bars" in q:
            self._one = {"min_ts": self.now - timedelta(hours=2), "max_ts": self.now}
            return
        if "from feature_snapshots_main" in q and "schema_hash_variants" in q:
            self._one = {
                "rows_total": 100,
                "schema_hash_variants": 1,
                "schema_hash_mismatch": 0,
                "feature_dim_variants": 1,
                "feature_dim_ok_rows": 100,
                "min_ts": self.now - timedelta(hours=2),
                "max_ts": self.now,
            }
            return
        if "from feature_snapshots_main" in q and "bucket_count" in q:
            self._one = {"bucket_count": 60}
            return
        if "from feature_matrix_main" in q and "vector_dim_ok_rows" in q:
            self._one = {
                "rows_total": 100,
                "feature_dim_ok_rows": 100,
                "vector_dim_ok_rows": 100,
                "schema_hash_mismatch": 0,
                "schema_hash_variants": 1,
                "min_ts": self.now - timedelta(hours=2),
                "max_ts": self.now,
            }
            return
        if "from feature_matrix_main" in q and "bucket_count" in q:
            self._one = {"bucket_count": 60}
            return
        if "select max(ts) as max_ts from market_bars" in q:
            self._one = {"max_ts": self.now}
            return
        if "select max(as_of_ts) as max_ts from feature_snapshots_main" in q:
            self._one = {"max_ts": self.now}
            return
        if "select max(as_of_ts) as max_ts from feature_matrix_main" in q:
            self._one = {"max_ts": self.now}
            return
        if "select count(*)::bigint as c from market_bars" in q:
            self._one = {"c": 9999}
            return
        if "from asset_universe_snapshots" in q:
            self._one = {
                "track": "liquid",
                "as_of": self.now - timedelta(days=1),
                "universe_version": "u1",
                "source": "seed",
                "symbols_json": {"symbols": ["BTC", "ETH"]},
                "created_at": self.now - timedelta(days=1),
            }
            return
        if "insert into offline_data_audits" in q:
            self._one = {}
            return
        if "select count(*)::bigint as c from social_" in q or "from events" in q:
            self._one = {"c": 0}
            return
        self._one = {}

    def fetchone(self):
        if self._rows:
            return self._rows[0]
        return self._one

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def cursor(self):
        return _FakeCursor()

    def commit(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_offline_data_audit_runs_without_social_tables(monkeypatch):
    monkeypatch.setattr(audit_mod.psycopg2, "connect", lambda *args, **kwargs: _FakeConn())
    now = datetime.now(timezone.utc)
    args = audit_mod.AuditArgs(
        database_url="postgresql://unused",
        track="liquid",
        symbols=["BTC", "ETH"],
        start=now - timedelta(hours=5),
        end=now,
        lookback=96,
        bucket="5m",
        min_market_ratio=0.5,
        min_feature_matrix_ratio=0.5,
        strict_schema_mismatch_zero=True,
        min_text_coverage=0.05,
        enforce_text_coverage=False,
        created_by="pytest",
        task_id="task-audit-minimal",
    )
    out = audit_mod.run_audit(args)
    assert isinstance(out, dict)
    assert out.get("task_id") == "task-audit-minimal"
    assert "coverage" in out
    assert "gates" in out
    assert out["coverage"]["text_source"]["social_posts_raw"]["exists"] is False
