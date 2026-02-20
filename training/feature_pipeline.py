from __future__ import annotations

import json
import os
import sys
import uuid
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
_ROOT = Path(__file__).resolve().parents[1]
_INFER_DIR = _ROOT / "inference"
if str(_INFER_DIR) not in sys.path:
    sys.path.append(str(_INFER_DIR))

LIQUID_FEATURE_SCHEMA_VERSION = os.getenv("FEATURE_PAYLOAD_SCHEMA_VERSION", "main")
ONCHAIN_FLOW_METRIC_NAMES: List[str] = [
    "netflow",
    "exchange_netflow",
    "net_inflow",
]
DERIVATIVE_METRIC_KEY_MAP: Dict[str, str] = {
    "long_short_ratio_global_accounts": "deriv_long_short_ratio_global_accounts",
    "long_short_ratio_top_accounts": "deriv_long_short_ratio_top_accounts",
    "long_short_ratio_top_positions": "deriv_long_short_ratio_top_positions",
    "taker_buy_sell_ratio": "deriv_taker_buy_sell_ratio",
    "basis_rate": "deriv_basis_rate",
    "annualized_basis_rate": "deriv_annualized_basis_rate",
}
DERIVATIVE_METRIC_NAMES: List[str] = list(DERIVATIVE_METRIC_KEY_MAP.keys())
DERIVATIVE_FEATURE_KEYS: List[str] = list(DERIVATIVE_METRIC_KEY_MAP.values())
DERIVATIVE_MISSING_FLAG_KEYS: List[str] = [f"{k}_missing_flag" for k in DERIVATIVE_FEATURE_KEYS]
LIQUID_FEATURE_KEYS: List[str] = [
    "ret_1",
    "ret_3",
    "ret_12",
    "ret_48",
    "ret_96",
    "ret_288",
    "vol_3",
    "vol_12",
    "vol_48",
    "vol_96",
    "vol_288",
    "ret_accel_1_3",
    "ret_accel_3_12",
    "ret_accel_12_48",
    "vol_term_3_12",
    "vol_term_12_48",
    "vol_term_48_288",
    "log_volume",
    "vol_z",
    "volume_impact",
    "orderbook_imbalance",
    "funding_rate",
    "onchain_norm",
    "deriv_long_short_ratio_global_accounts",
    "deriv_long_short_ratio_top_accounts",
    "deriv_long_short_ratio_top_positions",
    "deriv_taker_buy_sell_ratio",
    "deriv_basis_rate",
    "deriv_annualized_basis_rate",
    "event_decay",
    "orderbook_missing_flag",
    "funding_missing_flag",
    "onchain_missing_flag",
    "deriv_long_short_ratio_global_accounts_missing_flag",
    "deriv_long_short_ratio_top_accounts_missing_flag",
    "deriv_long_short_ratio_top_positions_missing_flag",
    "deriv_taker_buy_sell_ratio_missing_flag",
    "deriv_basis_rate_missing_flag",
    "deriv_annualized_basis_rate_missing_flag",
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
    "event_density",
    "sentiment_abs",
    "social_comment_rate",
    "event_importance_mean",
    "novelty_confidence_blend",
]
ONLINE_FEATURE_KEYS: List[str] = list(LIQUID_FEATURE_KEYS)
try:
    from liquid_feature_contract import LIQUID_FEATURE_KEYS as _CONTRACT_KEYS  # type: ignore
    from liquid_feature_contract import LIQUID_FEATURE_SCHEMA_VERSION as _CONTRACT_SCHEMA  # type: ignore
    from liquid_feature_contract import ONLINE_LIQUID_FEATURE_KEYS as _ONLINE_KEYS  # type: ignore
    from liquid_feature_contract import DERIVATIVE_METRIC_KEY_MAP as _CONTRACT_DERIVATIVE_METRIC_KEY_MAP  # type: ignore
    from liquid_feature_contract import DERIVATIVE_METRIC_NAMES as _CONTRACT_DERIVATIVE_METRIC_NAMES  # type: ignore
    from liquid_feature_contract import ONCHAIN_FLOW_METRIC_NAMES as _CONTRACT_ONCHAIN_FLOW_METRIC_NAMES  # type: ignore

    if isinstance(_CONTRACT_KEYS, list) and _CONTRACT_KEYS:
        LIQUID_FEATURE_KEYS = list(_CONTRACT_KEYS)
    if isinstance(_ONLINE_KEYS, list) and _ONLINE_KEYS:
        ONLINE_FEATURE_KEYS = list(_ONLINE_KEYS)
    if isinstance(_CONTRACT_DERIVATIVE_METRIC_KEY_MAP, dict) and _CONTRACT_DERIVATIVE_METRIC_KEY_MAP:
        DERIVATIVE_METRIC_KEY_MAP = {str(k): str(v) for k, v in _CONTRACT_DERIVATIVE_METRIC_KEY_MAP.items()}
        DERIVATIVE_METRIC_NAMES = list(DERIVATIVE_METRIC_KEY_MAP.keys())
        DERIVATIVE_FEATURE_KEYS = list(DERIVATIVE_METRIC_KEY_MAP.values())
        DERIVATIVE_MISSING_FLAG_KEYS = [f"{k}_missing_flag" for k in DERIVATIVE_FEATURE_KEYS]
    if isinstance(_CONTRACT_ONCHAIN_FLOW_METRIC_NAMES, list) and _CONTRACT_ONCHAIN_FLOW_METRIC_NAMES:
        ONCHAIN_FLOW_METRIC_NAMES = [str(x) for x in _CONTRACT_ONCHAIN_FLOW_METRIC_NAMES if str(x).strip()]
    LIQUID_FEATURE_SCHEMA_VERSION = str(_CONTRACT_SCHEMA or LIQUID_FEATURE_SCHEMA_VERSION)
except Exception:
    pass


@dataclass
class SampleBatch:
    X: np.ndarray
    y: np.ndarray
    meta: List[Dict]
    extra_labels: Dict[str, np.ndarray] | None = None
    sampling: Dict[str, object] | None = None


class FeaturePipeline:
    def __init__(self, db_url: str = DATABASE_URL):
        self.db_url = db_url

    def _connect(self):
        return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)

    @staticmethod
    def _table_exists(cur, table_name: str) -> bool:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = %s
            ) AS exists_flag
            """,
            (str(table_name).lower(),),
        )
        row = cur.fetchone() or {}
        return bool(row.get("exists_flag"))

    @staticmethod
    def _dq_thresholds() -> Dict[str, float]:
        return {
            "missing_rate_max": float(os.getenv("DQ_MAX_MISSING_RATE", "0.02")),
            "invalid_price_rate_max": float(os.getenv("DQ_MAX_INVALID_PRICE_RATE", "0.005")),
            "duplicate_rate_max": float(os.getenv("DQ_MAX_DUPLICATE_RATE", "0.02")),
            "stale_ratio_max": float(os.getenv("DQ_MAX_STALE_RATIO", "0.1")),
            "min_rows": float(os.getenv("DQ_MIN_ROWS", "200")),
        }

    @staticmethod
    def _source_tier_weights() -> Dict[int, float]:
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

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        tf = str(timeframe or "5m").strip().lower()
        try:
            if tf.endswith("m"):
                return max(1, int(tf[:-1] or "5"))
            if tf.endswith("h"):
                return max(1, int(tf[:-1] or "1")) * 60
            if tf.endswith("d"):
                return max(1, int(tf[:-1] or "1")) * 1440
            return max(1, int(tf))
        except Exception:
            return 5

    @staticmethod
    def vector_to_feature_payload(feature_row: np.ndarray | List[float]) -> Dict[str, float]:
        arr = np.array(feature_row, dtype=np.float64).reshape(-1)
        return {k: float(arr[idx]) if idx < arr.shape[0] else 0.0 for idx, k in enumerate(LIQUID_FEATURE_KEYS)}

    @staticmethod
    def project_to_online_payload(payload: Dict[str, float]) -> Dict[str, float]:
        return {k: float(payload.get(k, 0.0) or 0.0) for k in ONLINE_FEATURE_KEYS}

    @staticmethod
    def _align_feature_vector(values: List[float]) -> List[float]:
        target_dim = len(LIQUID_FEATURE_KEYS)
        row = [float(x) for x in values]
        if len(row) == target_dim:
            return row
        if len(row) > target_dim:
            return row[:target_dim]
        return row + [0.0] * (target_dim - len(row))

    @staticmethod
    def _parse_optional_utc(raw: object) -> Optional[datetime]:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            dt = raw
        else:
            text = str(raw or "").strip()
            if not text:
                return None
            text = text.replace(" ", "T")
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(text)
            except Exception:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _sampling_indices(n_rows: int, max_samples: int, sample_mode: str) -> np.ndarray:
        n = max(0, int(n_rows))
        k = max(0, int(max_samples))
        if n <= 0:
            return np.zeros((0,), dtype=np.int64)
        if k <= 0 or n <= k:
            return np.arange(n, dtype=np.int64)
        mode = str(sample_mode or "uniform").strip().lower()
        if mode in {"tail", "latest"}:
            return np.arange(n - k, n, dtype=np.int64)
        if mode in {"head", "earliest"}:
            return np.arange(0, k, dtype=np.int64)
        # Default to deterministic uniform spacing to avoid temporal bias.
        return np.linspace(0, n - 1, k).astype(np.int64)

    @staticmethod
    def _latest_before_with_missing(ts_list: List[datetime], values: List[float], ts: datetime) -> tuple[float, float]:
        if not ts_list:
            return 0.0, 1.0
        idx = bisect_right(ts_list, ts) - 1
        if idx < 0:
            return 0.0, 1.0
        return float(values[idx]), 0.0

    @staticmethod
    def _normalize_derivative_value(metric_name: str, raw_value: float) -> float:
        name = str(metric_name or "").strip().lower()
        v = float(raw_value)
        if name in {"long_short_ratio_global_accounts", "long_short_ratio_top_accounts", "long_short_ratio_top_positions", "taker_buy_sell_ratio"}:
            return float(np.tanh(np.log(max(1e-9, v))))
        if name == "basis_rate":
            return float(np.tanh(v * 25.0))
        if name == "annualized_basis_rate":
            return float(np.tanh(v * 5.0))
        return v

    @staticmethod
    def _weighted_std(values: List[float], weights: List[float]) -> float:
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

    @staticmethod
    def _window_rows(
        ts_list: List[datetime],
        rows: List[Dict[str, object]],
        start_ts: datetime,
        end_ts: datetime,
        max_count: int = 256,
    ) -> List[Dict[str, object]]:
        if not ts_list or not rows:
            return []
        left = bisect_left(ts_list, start_ts)
        right = bisect_right(ts_list, end_ts)
        if right <= left:
            return []
        sliced = rows[left:right]
        if len(sliced) > max_count:
            sliced = sliced[-max_count:]
        return sliced

    @staticmethod
    def _social_agg_window_profile(rows: List[Dict[str, object]], as_of_ts: datetime) -> Dict[str, float]:
        if not rows:
            return {}
        total_items = 0.0
        total_posts = 0.0
        total_comments = 0.0
        total_engagement = 0.0
        total_authors = 0.0
        post_sent_num = 0.0
        comment_sent_num = 0.0
        count_1h = 0.0
        count_6h = 0.0
        lookback_1h = as_of_ts - timedelta(hours=1)
        for row in rows:
            ts = row.get("_as_of_ts")
            item_count = max(0.0, float(row.get("item_count") or 0.0))
            post_count = max(0.0, float(row.get("post_count") or 0.0))
            comment_count = max(0.0, float(row.get("comment_count") or 0.0))
            engagement_sum = max(0.0, float(row.get("engagement_sum") or 0.0))
            unique_author_count = max(0.0, float(row.get("unique_author_count") or 0.0))
            post_sent = float(np.clip(float(row.get("post_sentiment_weighted") or 0.0), -1.0, 1.0))
            comment_sent = float(np.clip(float(row.get("comment_sentiment_weighted") or 0.0), -1.0, 1.0))
            w = max(1.0, item_count)
            total_items += item_count
            total_posts += post_count
            total_comments += comment_count
            total_engagement += engagement_sum
            total_authors += unique_author_count
            post_sent_num += w * post_sent
            comment_sent_num += w * comment_sent
            if isinstance(ts, datetime):
                if ts >= lookback_1h:
                    count_1h += item_count
                count_6h += item_count

        den = max(1.0, total_items)
        post_sent_mean = float(post_sent_num / den)
        comment_sent_mean = float(comment_sent_num / den)
        social_comment_rate = float(total_comments / max(1.0, total_posts))
        social_event_ratio = float(min(1.0, total_items / 72.0))
        engagement_norm = float(np.tanh(np.log1p(total_engagement / den) / 3.0))
        influence_norm = float(np.tanh(np.log1p(total_authors / max(1.0, len(rows))) / 2.0))
        buzz_norm = float(np.tanh(np.log1p(total_items) / 4.0))
        sentiment_abs = float(min(1.0, abs(0.5 * (post_sent_mean + comment_sent_mean))))
        event_density = float(min(1.0, total_items / 72.0))
        return {
            "social_post_sentiment": post_sent_mean,
            "social_comment_sentiment": comment_sent_mean,
            "social_engagement_norm": engagement_norm,
            "social_influence_norm": influence_norm,
            "social_event_ratio": social_event_ratio,
            "social_buzz": buzz_norm,
            "event_velocity_1h": float(np.tanh(count_1h / 12.0)),
            "event_velocity_6h": float(np.tanh(count_6h / 72.0)),
            "comment_skew": float(comment_sent_mean - post_sent_mean),
            "event_density": event_density,
            "sentiment_abs": sentiment_abs,
            "social_comment_rate": social_comment_rate,
        }

    @staticmethod
    def _blend_event_profile(base: Dict[str, float], override: Dict[str, float], alpha: float = 0.35) -> Dict[str, float]:
        if not override:
            return base
        out = dict(base)
        w = max(0.0, min(1.0, float(alpha)))
        for key, val in override.items():
            if key not in out:
                continue
            bv = float(out.get(key, 0.0) or 0.0)
            vv = float(val)
            if abs(bv) <= 1e-12:
                out[key] = vv
            else:
                out[key] = float((1.0 - w) * bv + w * vv)
        return out

    @staticmethod
    def _event_social_temporal_profile(ev_rows: List[Dict[str, object]], as_of_ts: datetime) -> Dict[str, float]:
        base = {
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
            "event_density": 0.0,
            "sentiment_abs": 0.0,
            "social_comment_rate": 0.0,
            "event_importance_mean": 0.0,
            "novelty_confidence_blend": 0.0,
        }
        if not ev_rows:
            return base

        lookback_start = as_of_ts - timedelta(hours=24)
        selected: List[Dict[str, object]] = []
        for evt in ev_rows:
            evt_ts = evt.get("timestamp")
            if not isinstance(evt_ts, datetime):
                continue
            # No-lookahead invariant: features at t may only use events available at/before t.
            if evt_ts > as_of_ts:
                continue
            if evt_ts <= lookback_start:
                continue
            selected.append(evt)
        if not selected:
            return base

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
        social_comment_count = 0.0

        lag_0_1h = 0.0
        lag_1_6h = 0.0
        lag_6_24h = 0.0
        mass_1h = 0.0
        mass_6h = 0.0
        importance_num = 0.0
        novelty_conf_num = 0.0

        event_sent_values: List[float] = []
        event_sent_weights: List[float] = []
        source_mass: Dict[str, float] = {}
        source_sent_num: Dict[str, float] = {}

        for evt in selected:
            evt_ts = evt["timestamp"]
            age_hours = max(0.0, float((as_of_ts - evt_ts).total_seconds()) / 3600.0)
            tier_weight = float(evt.get("tier_weight") or 0.0)
            confidence = float(evt.get("confidence") or 0.0)
            raw_confidence = float(evt.get("raw_confidence") or 0.0)
            event_importance = float(evt.get("event_importance") or raw_confidence)
            novelty_score = float(evt.get("novelty_score") or 0.0)
            ew = max(0.0, tier_weight * confidence)
            if ew <= 0:
                continue
            decay = float(np.exp(-age_hours / 12.0))
            num += ew * decay
            den += ew
            tier_sum += tier_weight
            conf_sum += raw_confidence
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

            post_sent = float(evt.get("post_sentiment") or 0.0)
            comment_sent = float(evt.get("comment_sentiment") or 0.0)
            event_sent = float(np.clip(0.5 * (post_sent + comment_sent), -1.0, 1.0))
            event_sent_values.append(event_sent)
            event_sent_weights.append(ew)

            source_key = str(evt.get("source_key") or "unknown").strip().lower() or "unknown"
            source_mass[source_key] = source_mass.get(source_key, 0.0) + ew
            source_sent_num[source_key] = source_sent_num.get(source_key, 0.0) + ew * event_sent
            importance_num += ew * float(np.clip(event_importance, 0.0, 1.0))
            novelty_conf_num += ew * float(np.clip(novelty_score, 0.0, 1.0)) * float(np.clip(raw_confidence, 0.0, 1.0))

            if bool(evt.get("is_social")):
                social_cnt += 1
                social_den += ew
                social_post += ew * post_sent
                social_comment += ew * comment_sent
                social_engage += ew * float(np.log1p(float(evt.get("engagement_score") or 0.0)))
                social_followers += ew * float(np.log1p(float(evt.get("author_followers") or 0.0)))
                social_comment_count += max(0.0, float(evt.get("n_comments") or 0.0) + float(evt.get("n_replies") or 0.0))

        if den <= 1e-9 or cnt <= 0:
            return base

        source_keys = sorted(source_mass.keys())
        src_weights = [float(source_mass[k]) for k in source_keys]
        src_sent_means = [float(source_sent_num[k] / max(1e-9, source_mass[k])) for k in source_keys]
        if len(src_weights) >= 2:
            probs = np.array(src_weights, dtype=np.float64) / max(1e-9, float(np.sum(src_weights)))
            src_entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
            source_diversity = float(src_entropy / max(1e-9, np.log(float(len(src_weights)))))
        else:
            source_diversity = 0.0
        source_std = FeaturePipeline._weighted_std(src_sent_means, src_weights) if src_weights else 0.0
        event_disagreement = float(min(1.0, FeaturePipeline._weighted_std(event_sent_values, event_sent_weights)))
        cross_source_consensus = float(max(0.0, 1.0 - min(1.0, source_std)))

        social_post_sentiment = float(social_post / max(1e-9, social_den)) if social_den > 0 else 0.0
        social_comment_sentiment = float(social_comment / max(1e-9, social_den)) if social_den > 0 else 0.0
        social_engagement_norm = float(np.tanh((social_engage / max(1e-9, social_den)) / 6.0)) if social_den > 0 else 0.0
        social_influence_norm = float(np.tanh((social_followers / max(1e-9, social_den)) / 14.0)) if social_den > 0 else 0.0
        social_event_ratio = float(social_cnt / max(1, cnt))
        social_buzz = float(np.tanh(social_den))
        event_density = float(min(1.0, cnt / 32.0))
        sentiment_abs = float(min(1.0, np.average(np.abs(np.array(event_sent_values, dtype=np.float64)), weights=np.array(event_sent_weights, dtype=np.float64)) if event_sent_values else 0.0))
        social_comment_rate = float(np.tanh(social_comment_count / max(1.0, social_cnt * 20.0))) if social_cnt > 0 else 0.0
        event_importance_mean = float(importance_num / max(1e-9, den))
        novelty_confidence_blend = float(novelty_conf_num / max(1e-9, den))

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
            "cross_source_consensus": cross_source_consensus,
            "comment_skew": float(social_comment_sentiment - social_post_sentiment),
            "event_lag_bucket_0_1h": float(lag_0_1h / max(1e-9, den)),
            "event_lag_bucket_1_6h": float(lag_1_6h / max(1e-9, den)),
            "event_lag_bucket_6_24h": float(lag_6_24h / max(1e-9, den)),
            "event_density": event_density,
            "sentiment_abs": sentiment_abs,
            "social_comment_rate": social_comment_rate,
            "event_importance_mean": event_importance_mean,
            "novelty_confidence_blend": novelty_confidence_blend,
        }

    def check_data_quality(
        self,
        symbol: str,
        timeframe: str = "5m",
        lookback_hours: int = 48,
        min_rows_override: Optional[int] = None,
    ) -> Dict[str, float]:
        thresholds = self._dq_thresholds()
        tf_minutes = self._timeframe_to_minutes(timeframe)
        expected_rows = max(1, int((max(1, lookback_hours) * 60) / max(1, tf_minutes)))
        required_rows = int(thresholds["min_rows"]) if min_rows_override is None else int(min_rows_override)
        if min_rows_override is None and tf_minutes > 5:
            ratio = float(os.getenv("DQ_MIN_COVERAGE_RATIO", "0.7"))
            adaptive_rows = max(24, int(expected_rows * max(0.1, ratio)))
            required_rows = min(int(thresholds["min_rows"]), adaptive_rows)
        stale_mult = float(os.getenv("DQ_STALE_GAP_MULTIPLIER", "3.0"))
        stale_gap_seconds = int(max(1, tf_minutes * 60 * stale_mult))
        source_used = "market_bars"
        with self._connect() as conn:
            with conn.cursor() as cur:
                if self._table_exists(cur, "market_bars"):
                    cur.execute(
                        """
                        SELECT
                          COUNT(*) AS total_rows,
                          COALESCE(SUM(CASE WHEN close IS NULL OR volume IS NULL THEN 1 ELSE 0 END), 0) AS missing_rows,
                          COALESCE(SUM(CASE WHEN close <= 0 THEN 1 ELSE 0 END), 0) AS invalid_price_rows,
                          COALESCE(SUM(
                            CASE
                              WHEN prev_ts IS NOT NULL AND ts = prev_ts THEN 1
                              ELSE 0
                            END
                          ), 0) AS duplicate_rows,
                          COALESCE(SUM(
                            CASE
                              WHEN prev_ts IS NOT NULL AND EXTRACT(EPOCH FROM (ts - prev_ts)) > %s THEN 1
                              ELSE 0
                            END
                          ), 0) AS stale_gap_rows
                        FROM (
                          SELECT
                            ts, close, volume,
                            LAG(ts) OVER (ORDER BY ts ASC) AS prev_ts
                          FROM market_bars
                          WHERE symbol = UPPER(%s)
                            AND timeframe = %s
                            AND ts > NOW() - make_interval(hours => %s)
                        ) s
                        """,
                        (stale_gap_seconds, symbol, timeframe, lookback_hours),
                    )
                    row = dict(cur.fetchone() or {})
                else:
                    row = {"total_rows": 0, "missing_rows": 0, "invalid_price_rows": 0, "duplicate_rows": 0, "stale_gap_rows": 0}
                if int(row.get("total_rows") or 0) <= 0:
                    if self._table_exists(cur, "prices"):
                        source_used = "prices_fallback"
                        fallback_gap = int(max(stale_gap_seconds, float(os.getenv("DQ_PRICE_STALE_GAP_SECONDS", "10800"))))
                        cur.execute(
                            """
                            SELECT
                              COUNT(*) AS total_rows,
                              COALESCE(SUM(CASE WHEN price IS NULL OR volume IS NULL THEN 1 ELSE 0 END), 0) AS missing_rows,
                              COALESCE(SUM(CASE WHEN price <= 0 THEN 1 ELSE 0 END), 0) AS invalid_price_rows,
                              COALESCE(SUM(
                                CASE
                                  WHEN prev_ts IS NOT NULL AND timestamp = prev_ts THEN 1
                                  ELSE 0
                                END
                              ), 0) AS duplicate_rows,
                              COALESCE(SUM(
                                CASE
                                  WHEN prev_ts IS NOT NULL AND EXTRACT(EPOCH FROM (timestamp - prev_ts)) > %s THEN 1
                                  ELSE 0
                                END
                              ), 0) AS stale_gap_rows
                            FROM (
                              SELECT
                                timestamp, price, volume,
                                LAG(timestamp) OVER (ORDER BY timestamp ASC) AS prev_ts
                              FROM prices
                              WHERE symbol = UPPER(%s)
                                AND timestamp > NOW() - make_interval(hours => %s)
                            ) s
                            """,
                            (fallback_gap, symbol, lookback_hours),
                        )
                        row = dict(cur.fetchone() or {})
                    else:
                        source_used = "none"
                        row = {"total_rows": 0, "missing_rows": 0, "invalid_price_rows": 0, "duplicate_rows": 0, "stale_gap_rows": 0}
        total = int(row.get("total_rows") or 0)
        missing = int(row.get("missing_rows") or 0)
        invalid = int(row.get("invalid_price_rows") or 0)
        dup = int(row.get("duplicate_rows") or 0)
        stale = int(row.get("stale_gap_rows") or 0)
        missing_rate = float(missing / max(1, total))
        invalid_rate = float(invalid / max(1, total))
        dup_rate = float(dup / max(1, total))
        stale_ratio = float(stale / max(1, total))
        passed = (
            total >= int(required_rows)
            and missing_rate <= thresholds["missing_rate_max"]
            and invalid_rate <= thresholds["invalid_price_rate_max"]
            and dup_rate <= thresholds["duplicate_rate_max"]
            and stale_ratio <= thresholds["stale_ratio_max"]
        )
        return {
            "total_rows": float(total),
            "required_rows": float(required_rows),
            "source_used": source_used,
            "fallback_used": bool(source_used == "prices_fallback"),
            "timeframe_used": str(timeframe),
            "missing_rate": missing_rate,
            "invalid_price_rate": invalid_rate,
            "duplicate_rate": dup_rate,
            "stale_ratio": stale_ratio,
            "quality_passed": float(1.0 if passed else 0.0),
        }

    def load_vc_training_batch(self, limit: int = 2000) -> SampleBatch:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT e.id, e.event_type, e.source_tier, e.confidence_score,
                           EXTRACT(EPOCH FROM (NOW() - e.occurred_at))/86400.0 AS recency_days
                    FROM events e
                    ORDER BY e.occurred_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            return SampleBatch(X=np.zeros((0, 5), dtype=np.float32), y=np.zeros((0,), dtype=np.float32), meta=[])

        event_map = {"funding": 1.0, "product": 0.7, "mna": 0.9, "regulatory": -0.8, "market": 0.2}
        feats = []
        labels = []
        for r in rows:
            et = r.get("event_type", "market")
            recency = float(r.get("recency_days") or 0.0)
            source_tier = float(r.get("source_tier") or 3)
            confidence = float(r.get("confidence_score") or 0.5)
            f = [event_map.get(et, 0.0), source_tier / 5.0, confidence, max(0.0, 1.0 - recency / 365.0), recency / 365.0]
            feats.append(f)
            labels.append(1.0 if et in {"funding", "mna"} else 0.0)

        return SampleBatch(
            X=np.array(feats, dtype=np.float32),
            y=np.array(labels, dtype=np.float32),
            meta=rows,
        )

    def load_liquid_training_batch(
        self,
        symbol: str,
        limit: int = 2000,
        timeframe: str = "5m",
        *,
        start: Optional[datetime | str] = None,
        end: Optional[datetime | str] = None,
        max_samples: int = 0,
        sample_mode: str = "uniform",
    ) -> SampleBatch:
        rows: List[Dict] = []
        orderbook_rows: List[Dict] = []
        funding_rows: List[Dict] = []
        onchain_rows: List[Dict] = []
        derivative_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in DERIVATIVE_METRIC_NAMES}
        event_rows: List[Dict] = []
        global_event_rows: List[Dict] = []
        social_bucket_rows: List[Dict[str, object]] = []
        effective_timeframe = str(timeframe)
        start_dt = self._parse_optional_utc(start)
        end_dt = self._parse_optional_utc(end)
        if start_dt and end_dt and end_dt < start_dt:
            start_dt, end_dt = end_dt, start_dt
        source_used = "none"
        source_fallback_used = False
        with self._connect() as conn:
            with conn.cursor() as cur:
                has_market_bars = self._table_exists(cur, "market_bars")
                has_orderbook = self._table_exists(cur, "orderbook_l2")
                has_funding = self._table_exists(cur, "funding_rates")
                has_onchain = self._table_exists(cur, "onchain_signals")
                has_events = self._table_exists(cur, "events")
                has_event_links = self._table_exists(cur, "event_links")
                has_entities = self._table_exists(cur, "entities")
                has_social_text_latent = self._table_exists(cur, "social_text_latent")
                has_prices = self._table_exists(cur, "prices")
                if has_market_bars:
                    params: List[object] = [symbol, timeframe]
                    sql = """
                        SELECT symbol, close::float AS price, volume::float AS volume, ts AS timestamp
                        FROM market_bars
                        WHERE symbol = UPPER(%s)
                          AND timeframe = %s
                    """
                    if isinstance(start_dt, datetime):
                        sql += " AND ts >= %s"
                        params.append(start_dt)
                    if isinstance(end_dt, datetime):
                        sql += " AND ts <= %s"
                        params.append(end_dt)
                    sql += " ORDER BY ts DESC"
                    if int(limit) > 0:
                        sql += " LIMIT %s"
                        params.append(int(limit))
                    cur.execute(sql, tuple(params))
                    rows = [dict(r) for r in cur.fetchall()]
                if rows:
                    source_used = "market_bars"
                if not rows and has_prices:
                    effective_timeframe = os.getenv("LIQUID_PRICE_FALLBACK_TIMEFRAME", str(timeframe))
                    params = [symbol]
                    sql = """
                        SELECT symbol, price::float AS price, volume::float AS volume, timestamp
                        FROM prices
                        WHERE symbol = UPPER(%s)
                    """
                    if isinstance(start_dt, datetime):
                        sql += " AND timestamp >= %s"
                        params.append(start_dt)
                    if isinstance(end_dt, datetime):
                        sql += " AND timestamp <= %s"
                        params.append(end_dt)
                    sql += " ORDER BY timestamp DESC"
                    if int(limit) > 0:
                        sql += " LIMIT %s"
                        params.append(int(limit))
                    cur.execute(sql, tuple(params))
                    rows = [dict(r) for r in cur.fetchall()]
                    if rows:
                        source_used = "prices_fallback"
                        source_fallback_used = True
                rows = sorted(rows, key=lambda r: r["timestamp"])
                if rows:
                    for row in rows:
                        row["source_used"] = source_used
                        row["timeframe_used"] = str(effective_timeframe)
                        row["price_fallback_used"] = bool(source_fallback_used)
                if rows:
                    range_end = rows[-1]["timestamp"]
                    range_start = rows[0]["timestamp"] - timedelta(hours=48)
                    if has_orderbook:
                        cur.execute(
                            """
                            SELECT ts AS timestamp, imbalance::float AS imbalance
                            FROM orderbook_l2
                            WHERE symbol = UPPER(%s)
                              AND ts >= %s
                              AND ts <= %s
                            ORDER BY ts DESC
                            LIMIT %s
                            """,
                            (symbol, range_start, range_end, max(limit * 8, 2000)),
                        )
                        orderbook_rows = [dict(r) for r in cur.fetchall()]
                        orderbook_rows.sort(key=lambda r: r["timestamp"])
                    if has_funding:
                        cur.execute(
                            """
                            SELECT ts AS timestamp, funding_rate::float AS funding_rate
                            FROM funding_rates
                            WHERE symbol = UPPER(%s)
                              AND ts >= %s
                              AND ts <= %s
                            ORDER BY ts DESC
                            LIMIT %s
                            """,
                            (symbol, range_start, range_end, max(limit * 4, 1200)),
                        )
                        funding_rows = [dict(r) for r in cur.fetchall()]
                        funding_rows.sort(key=lambda r: r["timestamp"])
                    if has_onchain:
                        cur.execute(
                            """
                            SELECT ts AS timestamp, metric_name, metric_value::float AS metric_value
                            FROM onchain_signals
                            WHERE asset_symbol = UPPER(%s)
                              AND metric_name = ANY(%s)
                              AND ts >= %s
                              AND ts <= %s
                            ORDER BY ts DESC
                            LIMIT %s
                            """,
                            (
                                symbol,
                                [*ONCHAIN_FLOW_METRIC_NAMES, *DERIVATIVE_METRIC_NAMES],
                                range_start,
                                range_end,
                                max(limit * 12, 3600),
                            ),
                        )
                        raw_onchain_rows = [dict(r) for r in cur.fetchall()]
                        flow_rows = []
                        deriv_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in DERIVATIVE_METRIC_NAMES}
                        for rr in raw_onchain_rows:
                            metric_name = str(rr.get("metric_name") or "").strip().lower()
                            if metric_name in set(ONCHAIN_FLOW_METRIC_NAMES):
                                flow_rows.append(rr)
                                continue
                            if metric_name in deriv_rows:
                                deriv_rows[metric_name].append(rr)
                        onchain_rows = sorted(flow_rows, key=lambda r: r["timestamp"])
                        derivative_rows = {k: sorted(v, key=lambda r: r["timestamp"]) for k, v in deriv_rows.items()}
                    if has_events and has_event_links and has_entities:
                        try:
                            cur.execute(
                                """
                                SELECT
                                    e.id AS event_id,
                                    COALESCE(e.available_at, e.occurred_at) AS timestamp,
                                    e.source_tier,
                                    e.confidence_score,
                                    e.source_name,
                                    e.payload,
                                    1.0::double precision AS scope_weight
                                FROM events e
                                JOIN event_links el ON el.event_id = e.id
                                JOIN entities en ON en.id = el.entity_id
                                WHERE UPPER(en.symbol) = UPPER(%s)
                                  AND COALESCE(en.metadata->>'synthetic_link', 'false') <> 'true'
                                  AND COALESCE(e.available_at, e.occurred_at) >= %s
                                  AND COALESCE(e.available_at, e.occurred_at) <= %s
                                ORDER BY COALESCE(e.available_at, e.occurred_at) DESC
                                LIMIT %s
                                """,
                                (symbol, range_start, range_end, max(limit * 8, 2000)),
                            )
                        except Exception:
                            cur.execute(
                                """
                                SELECT
                                    e.id AS event_id,
                                    e.occurred_at AS timestamp,
                                    e.source_tier,
                                    e.confidence_score,
                                    e.source_name,
                                    e.payload,
                                    1.0::double precision AS scope_weight
                                FROM events e
                                JOIN event_links el ON el.event_id = e.id
                                JOIN entities en ON en.id = el.entity_id
                                WHERE UPPER(en.symbol) = UPPER(%s)
                                  AND COALESCE(en.metadata->>'synthetic_link', 'false') <> 'true'
                                  AND e.occurred_at >= %s
                                  AND e.occurred_at <= %s
                                ORDER BY e.occurred_at DESC
                                LIMIT %s
                                """,
                                (symbol, range_start, range_end, max(limit * 8, 2000)),
                            )
                        event_rows = [dict(r) for r in cur.fetchall()]
                        event_rows.sort(key=lambda r: r["timestamp"])
                    if has_events:
                        try:
                            cur.execute(
                                """
                                SELECT
                                    e.id AS event_id,
                                    COALESCE(e.available_at, e.occurred_at) AS timestamp,
                                    e.source_tier,
                                    e.confidence_score,
                                    e.source_name,
                                    e.payload,
                                    0.7::double precision AS scope_weight
                                FROM events e
                                WHERE (e.market_scope = 'macro'
                                   OR COALESCE(e.payload->>'global_impact', 'false') = 'true')
                                  AND COALESCE(e.available_at, e.occurred_at) >= %s
                                  AND COALESCE(e.available_at, e.occurred_at) <= %s
                                ORDER BY COALESCE(e.available_at, e.occurred_at) DESC
                                LIMIT %s
                                """,
                                (range_start, range_end, max(limit * 2, 600)),
                            )
                            global_event_rows = [dict(r) for r in cur.fetchall()]
                            global_event_rows.sort(key=lambda r: r["timestamp"])
                        except Exception:
                            global_event_rows = []
                    if has_social_text_latent:
                        try:
                            cur.execute(
                                """
                                SELECT as_of_ts, agg_features
                                FROM social_text_latent
                                WHERE as_of_ts >= %s
                                  AND as_of_ts <= %s
                                  AND UPPER(%s) = ANY(symbols)
                                ORDER BY as_of_ts ASC
                                LIMIT %s
                                """,
                                (range_start, range_end, symbol, max(limit * 8, 2000)),
                            )
                            social_bucket_rows = [
                                {
                                    "_as_of_ts": r.get("as_of_ts"),
                                    **(dict(r.get("agg_features") or {}) if isinstance(r.get("agg_features"), dict) else {}),
                                }
                                for r in (cur.fetchall() or [])
                            ]
                        except Exception:
                            social_bucket_rows = []

        tf_minutes = self._timeframe_to_minutes(effective_timeframe)
        step_1h = max(1, int(round(60.0 / max(1, tf_minutes))))
        step_4h = max(1, step_1h * 4)
        history_len = 96
        if len(rows) < (history_len + step_4h + 2):
            return SampleBatch(
                X=np.zeros((0, len(LIQUID_FEATURE_KEYS)), dtype=np.float32),
                y=np.zeros((0,), dtype=np.float32),
                meta=[],
                extra_labels={},
                sampling={
                    "symbol": str(symbol).upper(),
                    "timeframe": str(effective_timeframe),
                    "query_start": start_dt.isoformat().replace("+00:00", "Z") if isinstance(start_dt, datetime) else "",
                    "query_end": end_dt.isoformat().replace("+00:00", "Z") if isinstance(end_dt, datetime) else "",
                    "limit": int(limit),
                    "max_samples": int(max_samples),
                    "sample_mode": str(sample_mode),
                    "raw_rows": int(len(rows)),
                    "feature_rows_before_sampling": 0,
                    "feature_rows_after_sampling": 0,
                },
            )

        ob_ts = [r["timestamp"] for r in orderbook_rows]
        ob_vals = [float(r.get("imbalance") or 0.0) for r in orderbook_rows]
        fr_ts = [r["timestamp"] for r in funding_rows]
        fr_vals = [float(r.get("funding_rate") or 0.0) for r in funding_rows]
        onchain_by_ts: Dict[datetime, List[float]] = {}
        for rr in onchain_rows:
            ts_obj = rr.get("timestamp")
            if not isinstance(ts_obj, datetime):
                continue
            ts_key = ts_obj if ts_obj.tzinfo is not None else ts_obj.replace(tzinfo=timezone.utc)
            ts_key = ts_key.astimezone(timezone.utc)
            onchain_by_ts.setdefault(ts_key, []).append(float(rr.get("metric_value") or 0.0))
        oc_ts = sorted(onchain_by_ts.keys())
        oc_vals = [float(np.mean(onchain_by_ts[k])) for k in oc_ts]
        deriv_ts_map: Dict[str, List[datetime]] = {}
        deriv_vals_map: Dict[str, List[float]] = {}
        for metric_name in DERIVATIVE_METRIC_NAMES:
            ts_list: List[datetime] = []
            val_list: List[float] = []
            for rr in derivative_rows.get(metric_name, []):
                ts_obj = rr.get("timestamp")
                if not isinstance(ts_obj, datetime):
                    continue
                ts_cur = ts_obj if ts_obj.tzinfo is not None else ts_obj.replace(tzinfo=timezone.utc)
                ts_cur = ts_cur.astimezone(timezone.utc)
                ts_list.append(ts_cur)
                val_list.append(float(rr.get("metric_value") or 0.0))
            deriv_ts_map[metric_name] = ts_list
            deriv_vals_map[metric_name] = val_list
        min_event_conf = float(os.getenv("EVENT_MIN_CONFIDENCE", "0.0"))
        max_event_tier = int(os.getenv("EVENT_MAX_SOURCE_TIER", "5"))
        tier_weights = self._source_tier_weights()
        ev_rows: List[Dict[str, object]] = []
        seen_event_ids: set[int] = set()
        for r in event_rows + global_event_rows:
            evt_id = int(r.get("event_id") or 0)
            if evt_id > 0:
                if evt_id in seen_event_ids:
                    continue
                seen_event_ids.add(evt_id)
            tier = int(r.get("source_tier") or 5)
            conf = float(r.get("confidence_score") or 0.0)
            if conf < min_event_conf or tier > max_event_tier:
                continue
            payload = r.get("payload") if isinstance(r.get("payload"), dict) else {}
            social_platform = str(payload.get("social_platform") or "").strip().lower() if isinstance(payload, dict) else ""
            is_social = bool(social_platform and social_platform not in {"none", "unknown"})
            post_sent = float(payload.get("post_sentiment", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
            comment_sent = float(payload.get("comment_sentiment", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
            engagement = float(payload.get("engagement_score", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
            followers = float(payload.get("author_followers", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
            n_comments = float(payload.get("n_comments", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
            n_replies = float(payload.get("n_replies", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
            backfill_added = float(payload.get("comment_backfill_added", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
            real_comment_total = max(0.0, n_comments + n_replies - backfill_added)
            event_importance = float(r.get("event_importance") or payload.get("event_importance") or 0.0) if isinstance(payload, dict) else float(r.get("event_importance") or 0.0)
            novelty_score = float(r.get("novelty_score") or payload.get("novelty_score") or 0.0) if isinstance(payload, dict) else float(r.get("novelty_score") or 0.0)
            source_key = (
                str(r.get("source_name") or "").strip().lower()
                or social_platform
                or str(payload.get("source") or "").strip().lower()
                or "unknown"
            )
            ev_rows.append(
                {
                    "timestamp": r["timestamp"],
                    "tier": tier,
                    "raw_confidence": conf,
                    "confidence": conf * float(r.get("scope_weight") or 1.0),
                    "tier_weight": float(tier_weights.get(tier, 0.1)),
                    "is_social": is_social,
                    "post_sentiment": float(np.clip(post_sent, -1.0, 1.0)),
                    "comment_sentiment": float(np.clip(comment_sent, -1.0, 1.0)),
                    "engagement_score": max(0.0, engagement),
                    "author_followers": max(0.0, followers),
                    # Exclude synthetic comment backfill from model features.
                    "n_comments": float(real_comment_total),
                    "n_replies": 0.0,
                    "event_importance": float(np.clip(event_importance, 0.0, 1.0)),
                    "novelty_score": float(np.clip(novelty_score, 0.0, 1.0)),
                    "source_key": source_key,
                }
            )
        ev_rows.sort(key=lambda x: x["timestamp"])
        social_agg_blend_alpha = float(os.getenv("SOCIAL_AGG_BLEND_ALPHA", "0.35"))
        social_bucket_rows = sorted(
            [r for r in social_bucket_rows if isinstance(r.get("_as_of_ts"), datetime)],
            key=lambda x: x["_as_of_ts"],  # type: ignore[index]
        )
        social_bucket_ts = [r["_as_of_ts"] for r in social_bucket_rows] if social_bucket_rows else []

        feats = []
        labels = []
        labels_1h = []
        labels_4h = []
        labels_cost = []
        sample_meta: List[Dict] = []
        for i in range(history_len, len(rows) - step_4h):
            price = float(rows[i].get("price") or 0.0)
            if price <= 0:
                continue
            prev_1 = float(rows[i - 1].get("price") or price)
            prev_3 = float(rows[i - 3].get("price") or price)
            prev_12 = float(rows[i - 12].get("price") or price)
            prev_48 = float(rows[i - 48].get("price") or price)
            prev_96 = float(rows[i - 96].get("price") or price)
            prev_288 = float(rows[max(0, i - 288)].get("price") or prev_96)
            ret_1 = (price - prev_1) / max(prev_1, 1e-12)
            ret_3 = (price - prev_3) / max(prev_3, 1e-12)
            ret_12 = (price - prev_12) / max(prev_12, 1e-12)
            ret_48 = (price - prev_48) / max(prev_48, 1e-12)
            ret_96 = (price - prev_96) / max(prev_96, 1e-12)
            ret_288 = (price - prev_288) / max(prev_288, 1e-12)
            w3 = np.array([float(rows[j].get("price") or price) for j in range(i - 3, i)], dtype=np.float64)
            w12 = np.array([float(rows[j].get("price") or price) for j in range(i - 12, i)], dtype=np.float64)
            w48 = np.array([float(rows[j].get("price") or price) for j in range(i - 48, i)], dtype=np.float64)
            w96 = np.array([float(rows[j].get("price") or price) for j in range(i - 96, i)], dtype=np.float64)
            w288 = np.array([float(rows[j].get("price") or price) for j in range(max(0, i - 288), i)], dtype=np.float64)
            if w288.size < 3:
                w288 = np.array([price, price, price], dtype=np.float64)
            vol_3 = float(np.std(np.diff(np.log(np.clip(w3, 1e-12, None)))))
            vol_12 = float(np.std(np.diff(np.log(np.clip(w12, 1e-12, None)))))
            vol_48 = float(np.std(np.diff(np.log(np.clip(w48, 1e-12, None)))))
            vol_96 = float(np.std(np.diff(np.log(np.clip(w96, 1e-12, None)))))
            vol_288 = float(np.std(np.diff(np.log(np.clip(w288, 1e-12, None)))))
            ret_accel_1_3 = float(ret_1 - ret_3)
            ret_accel_3_12 = float(ret_3 - ret_12)
            ret_accel_12_48 = float(ret_12 - ret_48)
            vol_term_3_12 = float(vol_3 - vol_12)
            vol_term_12_48 = float(vol_12 - vol_48)
            vol_term_48_288 = float(vol_48 - vol_288)
            vol = float(rows[i].get("volume") or 0.0)
            vol_hist = np.array([float(rows[j].get("volume") or 0.0) for j in range(i - 12, i)], dtype=np.float64)
            vol_z = float((vol - np.mean(vol_hist)) / max(np.std(vol_hist), 1e-6))
            volume_impact = float(abs(ret_1) / max(np.sqrt(max(vol, 1.0)), 1e-6))
            ts = rows[i]["timestamp"]
            if isinstance(ts, datetime):
                ts = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
                ts = ts.astimezone(timezone.utc)
            orderbook_imbalance, orderbook_missing_flag = self._latest_before_with_missing(ob_ts, ob_vals, ts)
            funding, funding_missing_flag = self._latest_before_with_missing(fr_ts, fr_vals, ts)
            onchain_flow, onchain_missing_flag = self._latest_before_with_missing(oc_ts, oc_vals, ts)
            onchain_norm = float(np.tanh(onchain_flow / 1e6))
            deriv_values: Dict[str, float] = {}
            deriv_missing_flags: Dict[str, float] = {}
            for metric_name in DERIVATIVE_METRIC_NAMES:
                cur_raw, cur_missing = self._latest_before_with_missing(
                    deriv_ts_map.get(metric_name, []),
                    deriv_vals_map.get(metric_name, []),
                    ts,
                )
                deriv_values[metric_name] = self._normalize_derivative_value(metric_name, cur_raw)
                deriv_missing_flags[metric_name] = float(cur_missing)
            event_profile = self._event_social_temporal_profile(ev_rows, ts)
            if social_bucket_rows and social_bucket_ts:
                agg_rows = self._window_rows(
                    social_bucket_ts,
                    social_bucket_rows,
                    start_ts=ts - timedelta(hours=6),
                    end_ts=ts,
                    max_count=256,
                )
                agg_profile = self._social_agg_window_profile(agg_rows, as_of_ts=ts)
                event_profile = self._blend_event_profile(event_profile, agg_profile, alpha=social_agg_blend_alpha)
            base_row = [
                ret_1,
                ret_3,
                ret_12,
                ret_48,
                ret_96,
                ret_288,
                vol_3,
                vol_12,
                vol_48,
                vol_96,
                vol_288,
                ret_accel_1_3,
                ret_accel_3_12,
                ret_accel_12_48,
                vol_term_3_12,
                vol_term_12_48,
                vol_term_48_288,
                np.log1p(max(vol, 0.0)),
                vol_z,
                volume_impact,
                orderbook_imbalance,
                funding,
                onchain_norm,
                float(deriv_values["long_short_ratio_global_accounts"]),
                float(deriv_values["long_short_ratio_top_accounts"]),
                float(deriv_values["long_short_ratio_top_positions"]),
                float(deriv_values["taker_buy_sell_ratio"]),
                float(deriv_values["basis_rate"]),
                float(deriv_values["annualized_basis_rate"]),
                float(event_profile["event_decay"]),
                orderbook_missing_flag,
                funding_missing_flag,
                onchain_missing_flag,
                float(deriv_missing_flags["long_short_ratio_global_accounts"]),
                float(deriv_missing_flags["long_short_ratio_top_accounts"]),
                float(deriv_missing_flags["long_short_ratio_top_positions"]),
                float(deriv_missing_flags["taker_buy_sell_ratio"]),
                float(deriv_missing_flags["basis_rate"]),
                float(deriv_missing_flags["annualized_basis_rate"]),
                float(event_profile["source_tier_weight"]),
                float(event_profile["source_confidence"]),
                float(event_profile["social_post_sentiment"]),
                float(event_profile["social_comment_sentiment"]),
                float(event_profile["social_engagement_norm"]),
                float(event_profile["social_influence_norm"]),
                float(event_profile["social_event_ratio"]),
                float(event_profile["social_buzz"]),
                float(event_profile["event_velocity_1h"]),
                float(event_profile["event_velocity_6h"]),
                float(event_profile["event_disagreement"]),
                float(event_profile["source_diversity"]),
                float(event_profile["cross_source_consensus"]),
                float(event_profile["comment_skew"]),
                float(event_profile["event_lag_bucket_0_1h"]),
                float(event_profile["event_lag_bucket_1_6h"]),
                float(event_profile["event_lag_bucket_6_24h"]),
                float(event_profile["event_density"]),
                float(event_profile["sentiment_abs"]),
                float(event_profile["social_comment_rate"]),
                float(event_profile["event_importance_mean"]),
                float(event_profile["novelty_confidence_blend"]),
            ]
            feats.append(self._align_feature_vector(base_row))

            fwd_1h = (float(rows[i + step_1h].get("price") or price) - price) / max(price, 1e-12)
            fwd_4h = (float(rows[i + step_4h].get("price") or price) - price) / max(price, 1e-12)
            est_cost = (5.0 + 3.0) / 10000.0 + min(0.002, 0.5 * abs(orderbook_imbalance) / 1000.0)
            labels.append(fwd_1h - est_cost)
            labels_1h.append(fwd_1h - est_cost)
            labels_4h.append(fwd_4h - est_cost * 2.0)
            labels_cost.append(est_cost)
            sample_meta.append(
                {
                    "as_of_ts": ts,
                    "as_of_ts_iso": ts.isoformat().replace("+00:00", "Z"),
                    "source_used": str(rows[i].get("source_used") or source_used),
                    "timeframe_used": str(rows[i].get("timeframe_used") or effective_timeframe),
                    "price_fallback_used": bool(rows[i].get("price_fallback_used") or source_fallback_used),
                }
            )

        feat_rows_before = int(len(feats))
        max_k = max(0, int(max_samples))
        mode = str(sample_mode or "uniform").strip().lower() or "uniform"
        if max_k > 0 and feat_rows_before > max_k:
            idx = self._sampling_indices(feat_rows_before, max_k, mode)
            idx_list = idx.tolist()
            feats = [feats[j] for j in idx_list]
            labels = [labels[j] for j in idx_list]
            labels_1h = [labels_1h[j] for j in idx_list]
            labels_4h = [labels_4h[j] for j in idx_list]
            labels_cost = [labels_cost[j] for j in idx_list]
            sample_meta = [sample_meta[j] for j in idx_list]

        return SampleBatch(
            X=np.array(feats, dtype=np.float32),
            y=np.array(labels, dtype=np.float32),
            meta=sample_meta,
            extra_labels={
                "fwd_ret_1h": np.array(labels_1h, dtype=np.float32),
                "fwd_ret_4h": np.array(labels_4h, dtype=np.float32),
                "est_cost": np.array(labels_cost, dtype=np.float32),
            },
            sampling={
                "symbol": str(symbol).upper(),
                "timeframe": str(effective_timeframe),
                "query_start": start_dt.isoformat().replace("+00:00", "Z") if isinstance(start_dt, datetime) else "",
                "query_end": end_dt.isoformat().replace("+00:00", "Z") if isinstance(end_dt, datetime) else "",
                "limit": int(limit),
                "max_samples": int(max_samples),
                "sample_mode": mode,
                "raw_rows": int(len(rows)),
                "feature_rows_before_sampling": feat_rows_before,
                "feature_rows_after_sampling": int(len(feats)),
                "source_used": source_used,
                "source_fallback_used": bool(source_fallback_used),
            },
        )

    def save_feature_snapshot(
        self,
        target: str,
        track: str,
        features: Dict[str, float],
        version: str = "feature-store-v2.0",
        lineage_id: Optional[str] = None,
        data_version: str = "v1",
        event_time: Optional[datetime] = None,
    ) -> str:
        lineage_id = lineage_id or uuid.uuid4().hex[:24]
        now = datetime.utcnow()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feature_snapshots (
                        target, track, as_of, as_of_ts, event_time, feature_available_at, feature_version,
                        feature_payload, data_version, lineage_id, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (target, track, now, now, event_time or now, now, version, json.dumps(features), data_version, lineage_id),
                )
        return lineage_id

    def save_feature_snapshots_bulk(
        self,
        target: str,
        track: str,
        feature_rows: List[Dict[str, float]],
        version: str = "feature-store-v2.0",
        lineage_id: Optional[str] = None,
        data_version: str = "v1",
        event_time: Optional[datetime] = None,
        row_times: Optional[List[datetime]] = None,
    ) -> str:
        lineage_id = lineage_id or uuid.uuid4().hex[:24]
        if not feature_rows:
            return lineage_id
        use_row_times = list(row_times or [])
        rows = []
        for idx, f in enumerate(feature_rows):
            base_ts = None
            if idx < len(use_row_times) and isinstance(use_row_times[idx], datetime):
                base_ts = use_row_times[idx]
            if base_ts is None:
                maybe_as_of = f.get("as_of_ts") if isinstance(f, dict) else None
                if isinstance(maybe_as_of, datetime):
                    base_ts = maybe_as_of
                elif isinstance(maybe_as_of, str) and maybe_as_of.strip():
                    try:
                        text = maybe_as_of.strip().replace(" ", "T")
                        if text.endswith("Z"):
                            text = text[:-1] + "+00:00"
                        base_ts = datetime.fromisoformat(text)
                    except Exception:
                        base_ts = None
            if base_ts is None and isinstance(event_time, datetime):
                base_ts = event_time
            if base_ts is None:
                raise ValueError("save_feature_snapshots_bulk requires per-sample row_times/as_of_ts")
            if base_ts.tzinfo is None:
                base_ts = base_ts.replace(tzinfo=timezone.utc)
            base_ts = base_ts.astimezone(timezone.utc)
            rows.append(
                (
                    target,
                    track,
                    base_ts,
                    base_ts,
                    base_ts,
                    base_ts,
                    version,
                    json.dumps(f),
                    data_version,
                    lineage_id,
                )
            )
        with self._connect() as conn:
            with conn.cursor() as cur:
                try:
                    execute_values(
                        cur,
                        """
                        INSERT INTO feature_snapshots (
                            target, track, as_of, as_of_ts, event_time, feature_available_at, feature_version,
                            feature_payload, data_version, lineage_id, created_at
                        ) VALUES %s
                        """,
                        rows,
                        template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
                    )
                except Exception:
                    rows_old = [r[:5] + r[6:] for r in rows]
                    execute_values(
                        cur,
                        """
                        INSERT INTO feature_snapshots (
                            target, track, as_of, as_of_ts, event_time, feature_version,
                            feature_payload, data_version, lineage_id, created_at
                        ) VALUES %s
                        """,
                        rows_old,
                        template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
                    )
        return lineage_id
