from __future__ import annotations

from typing import Any, Dict


def feature_signal_score(payload: Dict[str, Any]) -> float:
    if not isinstance(payload, dict):
        return 0.0
    ret_1 = float(payload.get("ret_1", 0.0) or 0.0)
    ret_3 = float(payload.get("ret_3", 0.0) or 0.0)
    ret_12 = float(payload.get("ret_12", 0.0) or 0.0)
    ret_48 = float(payload.get("ret_48", 0.0) or 0.0)
    vol_12 = float(payload.get("vol_12", 0.0) or 0.0)
    vol_48 = float(payload.get("vol_48", 0.0) or 0.0)
    vol_96 = float(payload.get("vol_96", 0.0) or 0.0)
    ob = float(payload.get("orderbook_imbalance", 0.0) or 0.0)
    funding = float(payload.get("funding_rate", 0.0) or 0.0)
    onchain = float(payload.get("onchain_norm", 0.0) or 0.0)
    event_decay = float(payload.get("event_decay", 0.0) or 0.0)
    source_tier_weight = float(payload.get("source_tier_weight", 0.0) or 0.0)
    source_confidence = float(payload.get("source_confidence", 0.0) or 0.0)
    social_post_sentiment = float(payload.get("social_post_sentiment", 0.0) or 0.0)
    social_comment_sentiment = float(payload.get("social_comment_sentiment", 0.0) or 0.0)
    social_engagement_norm = float(payload.get("social_engagement_norm", 0.0) or 0.0)
    social_influence_norm = float(payload.get("social_influence_norm", 0.0) or 0.0)
    social_event_ratio = float(payload.get("social_event_ratio", 0.0) or 0.0)
    social_buzz = float(payload.get("social_buzz", 0.0) or 0.0)
    vol_penalty = abs(vol_12) + abs(vol_48) + abs(vol_96)
    trend_bias = 0.6 * ret_12 + 0.4 * ret_48
    score = (
        0.10 * ret_1
        + 0.20 * ret_3
        + 0.50 * ret_12
        + 0.30 * ret_48
        + 0.35 * trend_bias
        + 0.10 * ob
        + 0.08 * funding
        + 0.07 * onchain
        + 0.05 * event_decay
        + 0.04 * source_tier_weight
        + 0.03 * source_confidence
        + 0.04 * social_post_sentiment
        + 0.03 * social_comment_sentiment
        + 0.02 * social_engagement_norm
        + 0.01 * social_influence_norm
        + 0.02 * social_event_ratio
        + 0.02 * social_buzz
        - 0.01 * vol_penalty
    )
    return float(score)
