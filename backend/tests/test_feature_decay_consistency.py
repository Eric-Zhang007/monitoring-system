from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math


def _decay(event_ts: datetime, as_of_ts: datetime) -> float:
    age_hours = max(0.0, (as_of_ts - event_ts).total_seconds() / 3600.0)
    return float(math.exp(-age_hours / 12.0))


def test_event_decay_formula_matches_training_and_inference_target():
    evt = datetime.now(timezone.utc) - timedelta(hours=3)
    as_of = datetime.now(timezone.utc)
    train_decay = _decay(evt, as_of)
    infer_decay = _decay(evt, as_of)
    assert abs(train_decay - infer_decay) < 1e-6
