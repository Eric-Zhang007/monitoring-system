from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

LIQUID_FEATURE_SCHEMA_VERSION = os.getenv("FEATURE_PAYLOAD_SCHEMA_VERSION", "v2.3")
LIQUID_FEATURE_KEYS: List[str] = [
    "ret_1",
    "ret_3",
    "ret_12",
    "ret_48",
    "vol_3",
    "vol_12",
    "vol_48",
    "vol_96",
    "log_volume",
    "vol_z",
    "volume_impact",
    "orderbook_imbalance",
    "funding_rate",
    "onchain_norm",
    "event_decay",
    "orderbook_missing_flag",
    "funding_missing_flag",
    "onchain_missing_flag",
    "source_tier_weight",
    "source_confidence",
    "social_post_sentiment",
    "social_comment_sentiment",
    "social_engagement_norm",
    "social_influence_norm",
    "social_event_ratio",
    "social_buzz",
    "event_velocity_1h",
    "event_velocity_6h",
    "event_disagreement",
    "source_diversity",
    "cross_source_consensus",
    "comment_skew",
    "event_lag_bucket_0_1h",
    "event_lag_bucket_1_6h",
    "event_lag_bucket_6_24h",
]


def source_tier_weights() -> Dict[int, float]:
    raw = os.getenv("SOURCE_TIER_WEIGHTS", "1=1.0,2=0.85,3=0.65,4=0.4,5=0.2")
    out: Dict[int, float] = {1: 1.0, 2: 0.85, 3: 0.65, 4: 0.4, 5: 0.2}
    for part in raw.split(","):
        piece = part.strip()
        if not piece or "=" not in piece:
            continue
        k_raw, v_raw = piece.split("=", 1)
        try:
            k = int(k_raw.strip())
            v = float(v_raw.strip())
        except Exception:
            continue
        if 1 <= k <= 5 and v >= 0:
            out[k] = v
    return out


def vector_from_payload(payload: Dict[str, float]) -> np.ndarray:
    return np.array([float(payload.get(k, 0.0) or 0.0) for k in LIQUID_FEATURE_KEYS], dtype=np.float32)


def weighted_std(values: List[float], weights: List[float]) -> float:
    if not values or not weights:
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
    max_tier = int(os.getenv("EVENT_MAX_SOURCE_TIER", "5"))
    min_conf = float(os.getenv("EVENT_MIN_CONFIDENCE", "0.0"))
    tier_weights = source_tier_weights()
    now = as_of_ts or datetime.now(timezone.utc)
    accepted: List[Dict[str, object]] = []
    tier_counts: Dict[str, int] = {}
    seen_keys: set[str] = set()
    for e in event_ctx:
        evt_id = int(e.get("id") or 0)
        evt_key = f"id:{evt_id}" if evt_id > 0 else f"ts:{e.get('available_at') or e.get('occurred_at')}:{e.get('source_url')}"
        if evt_key in seen_keys:
            continue
        seen_keys.add(evt_key)
        tier = int(e.get("source_tier") or 5)
        conf = float(e.get("confidence_score") or 0.0)
        if tier > max_tier or conf < min_conf:
            continue
        ts = e.get("available_at") or e.get("occurred_at")
        if isinstance(ts, datetime):
            ts_utc = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        else:
            ts_utc = now
        ts_utc = ts_utc.astimezone(timezone.utc)
        if ts_utc > now:
            continue
        age_hours = max(0.0, (now - ts_utc).total_seconds() / 3600.0)
        if age_hours > 24.0:
            continue
        tier_w = float(tier_weights.get(tier, 0.1))
        payload = e.get("payload") if isinstance(e.get("payload"), dict) else {}
        social_platform = str(payload.get("social_platform") or "").strip().lower() if isinstance(payload, dict) else ""
        is_social = bool(social_platform and social_platform not in {"none", "unknown"})
        post_sent = float(payload.get("post_sentiment", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
        comment_sent = float(payload.get("comment_sentiment", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
        engagement = float(payload.get("engagement_score", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
        followers = float(payload.get("author_followers", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
        source_key = (
            str(e.get("source_name") or "").strip().lower()
            or social_platform
            or str(payload.get("source") or "").strip().lower()
            or "unknown"
        )
        accepted.append(
            {
                "age_hours": age_hours,
                "tier_weight": tier_w,
                "raw_confidence": conf,
                "confidence": conf,
                "is_social": 1.0 if is_social else 0.0,
                "post_sentiment": float(np.clip(post_sent, -1.0, 1.0)),
                "comment_sentiment": float(np.clip(comment_sent, -1.0, 1.0)),
                "engagement_score": max(0.0, engagement),
                "author_followers": max(0.0, followers),
                "source_key": source_key,
            }
        )
        tier_key = str(tier)
        tier_counts[tier_key] = tier_counts.get(tier_key, 0) + 1
    if not accepted:
        return {
            "event_decay": 0.0,
            "source_tier_weight": 0.0,
            "source_confidence": 0.0,
            "social_post_sentiment": 0.0,
            "social_comment_sentiment": 0.0,
            "social_engagement_norm": 0.0,
            "social_influence_norm": 0.0,
            "social_event_ratio": 0.0,
            "social_buzz": 0.0,
            "event_velocity_1h": 0.0,
            "event_velocity_6h": 0.0,
            "event_disagreement": 0.0,
            "source_diversity": 0.0,
            "cross_source_consensus": 0.0,
            "comment_skew": 0.0,
            "event_lag_bucket_0_1h": 0.0,
            "event_lag_bucket_1_6h": 0.0,
            "event_lag_bucket_6_24h": 0.0,
            "source_tiers": {},
            "missing_markers": ["event_quality_unavailable"],
        }

    num = 0.0
    den = 0.0
    tier_sum = 0.0
    conf_sum = 0.0
    cnt = 0
    social_cnt = 0
    social_den = 0.0
    social_post = 0.0
    social_comment = 0.0
    social_engage = 0.0
    social_followers = 0.0
    lag_0_1h = 0.0
    lag_1_6h = 0.0
    lag_6_24h = 0.0
    mass_1h = 0.0
    mass_6h = 0.0
    event_sent_values: List[float] = []
    event_sent_weights: List[float] = []
    source_mass: Dict[str, float] = {}
    source_sent_num: Dict[str, float] = {}
    for a in accepted:
        age_hours = float(a["age_hours"])
        decay = float(np.exp(-age_hours / 12.0))
        ew = max(0.0, float(a["tier_weight"]) * float(a["confidence"]))
        if ew <= 0:
            continue
        num += ew * decay
        den += ew
        tier_sum += float(a["tier_weight"])
        conf_sum += float(a["raw_confidence"])
        cnt += 1
        if age_hours <= 1.0:
            lag_0_1h += ew
            mass_1h += ew
        elif age_hours <= 6.0:
            lag_1_6h += ew
        else:
            lag_6_24h += ew
        if age_hours <= 6.0:
            mass_6h += ew

        post_sent = float(a.get("post_sentiment", 0.0) or 0.0)
        comment_sent = float(a.get("comment_sentiment", 0.0) or 0.0)
        event_sent = float(np.clip(0.5 * (post_sent + comment_sent), -1.0, 1.0))
        event_sent_values.append(event_sent)
        event_sent_weights.append(ew)

        source_key = str(a.get("source_key") or "unknown").strip().lower() or "unknown"
        source_mass[source_key] = source_mass.get(source_key, 0.0) + ew
        source_sent_num[source_key] = source_sent_num.get(source_key, 0.0) + ew * event_sent

        if float(a.get("is_social", 0.0)) > 0.5:
            social_cnt += 1
            social_den += ew
            social_post += ew * post_sent
            social_comment += ew * comment_sent
            social_engage += ew * float(np.log1p(float(a.get("engagement_score", 0.0))))
            social_followers += ew * float(np.log1p(float(a.get("author_followers", 0.0))))

    if den <= 1e-9 or cnt <= 0:
        return {
            "event_decay": 0.0,
            "source_tier_weight": 0.0,
            "source_confidence": 0.0,
            "social_post_sentiment": 0.0,
            "social_comment_sentiment": 0.0,
            "social_engagement_norm": 0.0,
            "social_influence_norm": 0.0,
            "social_event_ratio": 0.0,
            "social_buzz": 0.0,
            "event_velocity_1h": 0.0,
            "event_velocity_6h": 0.0,
            "event_disagreement": 0.0,
            "source_diversity": 0.0,
            "cross_source_consensus": 0.0,
            "comment_skew": 0.0,
            "event_lag_bucket_0_1h": 0.0,
            "event_lag_bucket_1_6h": 0.0,
            "event_lag_bucket_6_24h": 0.0,
            "source_tiers": {k: float(v) for k, v in sorted(tier_counts.items())},
            "missing_markers": ["event_quality_unavailable"],
        }

    source_keys = sorted(source_mass.keys())
    src_weights = [float(source_mass[k]) for k in source_keys]
    src_sent_means = [float(source_sent_num[k] / max(1e-9, source_mass[k])) for k in source_keys]
    if len(src_weights) >= 2:
        probs = np.array(src_weights, dtype=np.float64) / max(1e-9, float(np.sum(src_weights)))
        src_entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
        source_diversity = float(src_entropy / max(1e-9, np.log(float(len(src_weights)))))
    else:
        source_diversity = 0.0
    source_std = weighted_std(src_sent_means, src_weights) if src_weights else 0.0
    event_disagreement = float(min(1.0, weighted_std(event_sent_values, event_sent_weights)))

    social_post_sentiment = float(social_post / max(1e-9, social_den)) if social_den > 0 else 0.0
    social_comment_sentiment = float(social_comment / max(1e-9, social_den)) if social_den > 0 else 0.0
    social_engagement_norm = float(np.tanh((social_engage / max(1e-9, social_den)) / 6.0)) if social_den > 0 else 0.0
    social_influence_norm = float(np.tanh((social_followers / max(1e-9, social_den)) / 14.0)) if social_den > 0 else 0.0
    social_event_ratio = float(social_cnt / max(1, cnt))
    social_buzz = float(np.tanh(social_den))
    return {
        "event_decay": float(num / max(1e-9, den)),
        "source_tier_weight": float(tier_sum / max(1, cnt)),
        "source_confidence": float(conf_sum / max(1, cnt)),
        "social_post_sentiment": social_post_sentiment,
        "social_comment_sentiment": social_comment_sentiment,
        "social_engagement_norm": social_engagement_norm,
        "social_influence_norm": social_influence_norm,
        "social_event_ratio": social_event_ratio,
        "social_buzz": social_buzz,
        "event_velocity_1h": float(np.tanh(mass_1h)),
        "event_velocity_6h": float(np.tanh(mass_6h / 6.0)),
        "event_disagreement": event_disagreement,
        "source_diversity": source_diversity,
        "cross_source_consensus": float(max(0.0, 1.0 - min(1.0, source_std))),
        "comment_skew": float(social_comment_sentiment - social_post_sentiment),
        "event_lag_bucket_0_1h": float(lag_0_1h / max(1e-9, den)),
        "event_lag_bucket_1_6h": float(lag_1_6h / max(1e-9, den)),
        "event_lag_bucket_6_24h": float(lag_6_24h / max(1e-9, den)),
        "source_tiers": {k: float(v) for k, v in sorted(tier_counts.items())},
        "missing_markers": [],
    }
