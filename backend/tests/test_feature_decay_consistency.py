from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "training"))
sys.path.append(str(ROOT / "inference"))
sys.path.append(str(ROOT))

from feature_pipeline import FeaturePipeline, LIQUID_FEATURE_KEYS as TRAIN_FEATURE_KEYS, LIQUID_FEATURE_SCHEMA_VERSION as TRAIN_SCHEMA_VERSION  # noqa: E402
from liquid_feature_contract import LIQUID_FEATURE_KEYS as INFER_FEATURE_KEYS, LIQUID_FEATURE_SCHEMA_VERSION as INFER_SCHEMA_VERSION  # noqa: E402
from liquid_feature_contract import event_quality_profile as infer_event_quality_profile  # noqa: E402
from liquid_feature_contract import vector_from_payload as infer_vector_from_payload  # noqa: E402
from features.feature_contract import FEATURE_DIM, SCHEMA_HASH  # noqa: E402


def test_liquid_feature_contract_keys_and_schema_align_between_training_and_inference():
    assert list(TRAIN_FEATURE_KEYS) == list(INFER_FEATURE_KEYS)
    assert TRAIN_SCHEMA_VERSION == INFER_SCHEMA_VERSION
    assert str(TRAIN_SCHEMA_VERSION).strip() != ""
    assert len(TRAIN_FEATURE_KEYS) == len(set(TRAIN_FEATURE_KEYS))
    assert len(TRAIN_FEATURE_KEYS) == FEATURE_DIM
    assert len(TRAIN_FEATURE_KEYS) >= 100


def test_feature_vector_roundtrip_keeps_shape_and_order():
    payload = {k: float(i + 1) for i, k in enumerate(TRAIN_FEATURE_KEYS)}
    vec = infer_vector_from_payload(payload)
    assert int(vec.shape[0]) == len(TRAIN_FEATURE_KEYS)
    roundtrip = FeaturePipeline.vector_to_feature_payload(vec)
    for idx, key in enumerate(TRAIN_FEATURE_KEYS, start=1):
        assert float(roundtrip[key]) == float(idx)


def test_event_temporal_profile_matches_training_and_inference_and_ignores_future_events():
    as_of = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    events = [
        {
            "id": 1,
            "source_tier": 2,
            "confidence_score": 0.8,
            "source_name": "alpha",
            "occurred_at": as_of - timedelta(minutes=30),
            "available_at": as_of - timedelta(minutes=20),
            "payload": {
                "social_platform": "reddit",
                "post_sentiment": 0.6,
                "comment_sentiment": -0.2,
                "engagement_score": 120,
                "author_followers": 1000,
            },
        },
        {
            "id": 2,
            "source_tier": 3,
            "confidence_score": 0.7,
            "source_name": "beta",
            "occurred_at": as_of - timedelta(hours=4),
            "available_at": as_of - timedelta(hours=3, minutes=30),
            "payload": {
                "social_platform": "youtube",
                "post_sentiment": -0.3,
                "comment_sentiment": -0.1,
                "engagement_score": 40,
                "author_followers": 3000,
            },
        },
        {
            "id": 3,
            "source_tier": 1,
            "confidence_score": 0.9,
            "source_name": "future-feed",
            "occurred_at": as_of + timedelta(minutes=5),
            "available_at": as_of + timedelta(minutes=10),
            "payload": {
                "social_platform": "x",
                "post_sentiment": 1.0,
                "comment_sentiment": 1.0,
                "engagement_score": 9999,
                "author_followers": 500000,
            },
        },
    ]
    infer_profile = infer_event_quality_profile(events, as_of_ts=as_of)
    infer_profile_no_future = infer_event_quality_profile(events[:2], as_of_ts=as_of)
    for k in ("event_count_1h", "event_count_6h", "event_confidence_mean"):
        assert abs(float(infer_profile[k]) - float(infer_profile_no_future[k])) < 1e-8
    assert int(infer_profile["event_count_1h"]) == 2
    assert int(infer_profile["event_count_6h"]) == 2
    assert str(infer_profile["schema_hash"]) == SCHEMA_HASH
    assert int(infer_profile["feature_dim"]) == FEATURE_DIM
    assert isinstance(infer_profile["missing_markers"], list)
