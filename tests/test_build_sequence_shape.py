from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from features.feature_contract import FEATURE_DIM, SCHEMA_HASH
from features.sequence import build_sequence


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *_args, **_kwargs):
        return None

    def fetchall(self):
        return self.rows


class _FakeConn:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return _FakeCursor(self.rows)


def test_build_sequence_shape(monkeypatch):
    ts = datetime(2026, 2, 22, 0, 0, tzinfo=timezone.utc)
    rows = [
        {
            "as_of_ts": ts,
            "values": [0.0] * FEATURE_DIM,
            "mask": [0] * FEATURE_DIM,
            "features": {},
            "schema_hash": SCHEMA_HASH,
        }
    ]

    def _fake_connect(*_args, **_kwargs):
        return _FakeConn(rows)

    monkeypatch.setattr("features.sequence.psycopg2.connect", _fake_connect)

    out = build_sequence(db_url="postgresql://x", symbol="BTC", end_ts=ts, lookback=4)
    assert out.values.shape == (4, FEATURE_DIM)
    assert out.mask.shape == (4, FEATURE_DIM)
    assert out.schema_hash == SCHEMA_HASH
    assert np.all((out.mask == 0) | (out.mask == 1))
