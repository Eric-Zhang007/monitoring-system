from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from features.feature_contract import FEATURE_DIM, SCHEMA_HASH
from inference.feature_reader import fetch_sequence
from training.datasets.liquid_sequence_dataset import load_training_samples


class _FakeCursor:
    def __init__(self, state):
        self.state = state
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        text = str(sql)
        if "FROM market_bars" in text and "close::double precision AS close" in text:
            self._rows = list(self.state["market_rows"])
            return
        if "FROM feature_matrix_main" in text:
            symbol = str(params[0]).upper()
            start_ts = params[1]
            end_ts = params[2]
            out = []
            for r in self.state["feature_rows"]:
                if str(r["symbol"]).upper() != symbol:
                    continue
                if r["as_of_ts"] < start_ts or r["as_of_ts"] > end_ts:
                    continue
                out.append(dict(r))
            out.sort(key=lambda x: x["as_of_ts"])
            self._rows = out
            return
        self._rows = []

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self, *args, **kwargs):
        return _FakeCursor(self.state)


def _build_state():
    base = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    market_rows = []
    feature_rows = []
    for i in range(2300):
        ts = base + timedelta(minutes=5 * i)
        close = 100.0 + i * 0.1
        market_rows.append({"ts": ts, "close": close})

        values = np.full((FEATURE_DIM,), float(i), dtype=np.float32)
        mask = np.zeros((FEATURE_DIM,), dtype=np.uint8)
        if i % 7 == 0:
            # inject text missing for parity check
            values[-8:] = 0.0
            mask[-8:] = 1
        feature_rows.append(
            {
                "symbol": "BTC",
                "as_of_ts": ts,
                "values": [float(x) for x in values.tolist()],
                "mask": [int(x) for x in mask.tolist()],
                "features": {},
                "schema_hash": SCHEMA_HASH,
            }
        )
    return {"market_rows": market_rows, "feature_rows": feature_rows}


def test_train_infer_parity(monkeypatch):
    state = _build_state()

    def _fake_connect(*_args, **_kwargs):
        return _FakeConn(state)

    monkeypatch.setattr("training.datasets.liquid_sequence_dataset.psycopg2.connect", _fake_connect)
    monkeypatch.setattr("features.sequence.psycopg2.connect", _fake_connect)

    start = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 2, 8, 4, 0, tzinfo=timezone.utc)
    lookback = 16

    samples = load_training_samples(
        db_url="postgresql://x",
        symbols=["BTC"],
        start_ts=start,
        end_ts=end,
        lookback=lookback,
        max_samples_per_symbol=1,
    )
    assert samples
    sample = samples[0]

    infer = fetch_sequence(
        db_url="postgresql://x",
        symbol="BTC",
        end_ts=sample.end_ts,
        lookback=lookback,
    )

    assert sample.schema_hash == SCHEMA_HASH
    assert infer["schema_hash"] == SCHEMA_HASH
    assert np.array_equal(sample.x_values, infer["values"])
    assert np.array_equal(sample.x_mask, infer["mask"])
