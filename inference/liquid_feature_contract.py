from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from features.align import FeatureAlignmentError, align_row
from features.feature_contract import (
    FEATURE_DIM,
    FEATURE_KEYS,
    GROUP_MAP,
    SCHEMA_HASH,
    SCHEMA_VERSION,
)

# Backward-compatible exports: now all map to the single generated schema.
LIQUID_FEATURE_SCHEMA_VERSION = SCHEMA_VERSION
LIQUID_FEATURE_KEYS: List[str] = list(FEATURE_KEYS)
ONLINE_LIQUID_FEATURE_KEYS: List[str] = list(FEATURE_KEYS)

# Keep compatibility names used by older scripts.
LIQUID_MANUAL_FEATURE_KEYS: List[str] = [k for k in FEATURE_KEYS if GROUP_MAP.get(k) != "text"]
LIQUID_LATENT_FEATURE_KEYS: List[str] = [k for k in FEATURE_KEYS if GROUP_MAP.get(k) == "text"]
LIQUID_FULL_FEATURE_KEYS: List[str] = list(FEATURE_KEYS)

ONCHAIN_FLOW_METRIC_NAMES: List[str] = ["net_inflow", "netflow", "exchange_netflow"]
DEFAULT_ONCHAIN_PRIMARY_METRIC = "net_inflow"
DERIVATIVE_METRIC_KEY_MAP: Dict[str, str] = {
    "basis_rate": "basis_rate",
    "taker_buy_sell_ratio": "taker_buy_sell_ratio",
    "open_interest": "open_interest",
    "funding_rate": "funding_rate",
}
DERIVATIVE_METRIC_NAMES: List[str] = list(DERIVATIVE_METRIC_KEY_MAP.keys())
DERIVATIVE_FEATURE_KEYS: List[str] = list(DERIVATIVE_METRIC_KEY_MAP.values())
DERIVATIVE_MISSING_FLAG_KEYS: List[str] = []


def source_tier_weights() -> Dict[int, float]:
    return {1: 1.0, 2: 0.85, 3: 0.65, 4: 0.4, 5: 0.2}


def vector_from_payload(payload: Dict[str, float]) -> np.ndarray:
    aligned = align_row(payload)
    return aligned.values.astype(np.float32)


def project_to_online_schema(payload: Dict[str, float]) -> Dict[str, float]:
    # No online subset projection is allowed anymore.
    aligned = align_row(payload)
    return {k: float(aligned.values[idx]) for idx, k in enumerate(FEATURE_KEYS)}


def vector_from_payload_online(payload: Dict[str, float]) -> np.ndarray:
    # Kept for compatibility with old callers; now identical to vector_from_payload.
    return vector_from_payload(payload)


def weighted_std(values: List[float], weights: List[float]) -> float:
    if (not values) or (not weights):
        return 0.0
    vv = np.array(values, dtype=np.float64)
    ww = np.array(weights, dtype=np.float64)
    den = float(np.sum(ww))
    if den <= 1e-12:
        return 0.0
    mean = float(np.sum(vv * ww) / den)
    var = float(np.sum(ww * ((vv - mean) ** 2)) / den)
    return float(np.sqrt(max(0.0, var)))


def event_quality_profile(event_ctx: List[Dict], *, as_of_ts: Optional[datetime] = None) -> Dict[str, object]:
    now = as_of_ts or datetime.now(timezone.utc)
    cnt = 0
    conf = 0.0
    for e in event_ctx:
        ts = e.get("available_at") or e.get("occurred_at")
        if isinstance(ts, datetime):
            dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            if dt.astimezone(timezone.utc) > now:
                continue
        cnt += 1
        conf += float(e.get("confidence_score") or 0.0)
    mean_conf = conf / max(1, cnt)
    return {
        "event_count_1h": float(cnt),
        "event_count_6h": float(cnt),
        "event_confidence_mean": float(mean_conf),
        "event_sentiment_mean": 0.0,
        "source_tiers": {},
        "missing_markers": [],
        "schema_hash": SCHEMA_HASH,
        "feature_dim": FEATURE_DIM,
    }


__all__ = [
    "SCHEMA_HASH",
    "FEATURE_DIM",
    "LIQUID_FEATURE_SCHEMA_VERSION",
    "LIQUID_FEATURE_KEYS",
    "ONLINE_LIQUID_FEATURE_KEYS",
    "LIQUID_MANUAL_FEATURE_KEYS",
    "LIQUID_LATENT_FEATURE_KEYS",
    "LIQUID_FULL_FEATURE_KEYS",
    "ONCHAIN_FLOW_METRIC_NAMES",
    "DEFAULT_ONCHAIN_PRIMARY_METRIC",
    "DERIVATIVE_METRIC_KEY_MAP",
    "DERIVATIVE_METRIC_NAMES",
    "DERIVATIVE_FEATURE_KEYS",
    "DERIVATIVE_MISSING_FLAG_KEYS",
    "source_tier_weights",
    "vector_from_payload",
    "project_to_online_schema",
    "vector_from_payload_online",
    "weighted_std",
    "event_quality_profile",
    "FeatureAlignmentError",
]
