"""
Inference Service V2
- Dual-track prediction routing
- Prediction explanation persistence
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
import time

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import redis

from explainer import build_explanation
from model_router import ModelRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
LIQUID_SYMBOLS = [s.strip().upper() for s in os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK").split(",") if s.strip()]
INFER_INTERVAL_SEC = int(os.getenv("INFER_INTERVAL_SEC", "60"))


class InferenceServiceV2:
    def __init__(self):
        self.router = ModelRouter()
        self.redis_client = None
        self.conn = None
        self.active_model_cache: Dict[str, Dict[str, object]] = {}
        self.active_model_ttl_sec = int(os.getenv("ACTIVE_MODEL_TTL_SEC", "30"))
        self.feature_version = os.getenv("FEATURE_VERSION", "feature-store-v2.1")
        self.data_version = os.getenv("DATA_VERSION", "v1")

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

    def _event_quality_profile(self, event_ctx: List[Dict]) -> Dict[str, object]:
        max_tier = int(os.getenv("EVENT_MAX_SOURCE_TIER", "5"))
        min_conf = float(os.getenv("EVENT_MIN_CONFIDENCE", "0.0"))
        tier_weights = self._source_tier_weights()
        now = datetime.now(timezone.utc)
        accepted: List[Dict[str, float]] = []
        tier_counts: Dict[str, int] = {}
        for e in event_ctx:
            tier = int(e.get("source_tier") or 5)
            conf = float(e.get("confidence_score") or 0.0)
            if tier > max_tier or conf < min_conf:
                continue
            ts = e.get("available_at") or e.get("occurred_at")
            if isinstance(ts, datetime):
                ts_utc = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            else:
                ts_utc = now
            age_hours = max(0.0, (now - ts_utc.astimezone(timezone.utc)).total_seconds() / 3600.0)
            decay = float(np.exp(-age_hours / 12.0))
            tier_w = float(tier_weights.get(tier, 0.1))
            accepted.append(
                {
                    "decay": decay,
                    "tier_weight": tier_w,
                    "confidence": conf,
                    "joint_weight": tier_w * conf,
                }
            )
            tier_key = str(tier)
            tier_counts[tier_key] = tier_counts.get(tier_key, 0) + 1
        if not accepted:
            return {
                "event_decay": 0.0,
                "source_tier_weight": 0.0,
                "source_confidence": 0.0,
                "source_tiers": {},
                "missing_markers": ["event_quality_unavailable"],
            }
        den = float(sum(a["joint_weight"] for a in accepted))
        event_decay = float(sum(a["joint_weight"] * a["decay"] for a in accepted) / max(1e-9, den))
        tier_weight = float(sum(a["tier_weight"] for a in accepted) / max(1, len(accepted)))
        conf_mean = float(sum(a["confidence"] for a in accepted) / max(1, len(accepted)))
        return {
            "event_decay": event_decay,
            "source_tier_weight": tier_weight,
            "source_confidence": conf_mean,
            "source_tiers": {k: float(v) for k, v in sorted(tier_counts.items())},
            "missing_markers": [],
        }

    def connect(self):
        self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        self.conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT symbol, price::float AS price, volume::float AS volume, timestamp
                FROM prices
                WHERE symbol = UPPER(%s)
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (symbol,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        if not symbols:
            return {}
        with self.conn.cursor() as cur:
            cur.execute(
                """
                WITH ranked AS (
                    SELECT
                        symbol,
                        price::float AS price,
                        volume::float AS volume,
                        timestamp,
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
                    FROM prices
                    WHERE symbol = ANY(%s)
                )
                SELECT symbol, price, volume, timestamp
                FROM ranked
                WHERE rn = 1
                """,
                ([s.upper() for s in symbols],),
            )
            return {str(r["symbol"]).upper(): dict(r) for r in cur.fetchall()}

    def get_recent_prices(self, symbol: str, limit: int = 64) -> List[Dict]:
        with self.conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT symbol, close::float AS price, volume::float AS volume, ts AS timestamp
                    FROM market_bars
                    WHERE symbol = UPPER(%s) AND timeframe = '5m'
                    ORDER BY ts DESC
                    LIMIT %s
                    """,
                    (symbol, limit),
                )
                rows = [dict(r) for r in cur.fetchall()]
            except Exception:
                rows = []
            if rows:
                return sorted(rows, key=lambda x: x["timestamp"])

            cur.execute(
                """
                SELECT symbol, price::float AS price, volume::float AS volume, timestamp
                FROM prices
                WHERE symbol = UPPER(%s)
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (symbol, limit),
            )
            rows = [dict(r) for r in cur.fetchall()]
            return sorted(rows, key=lambda x: x["timestamp"])

    def get_orderbook_imbalance_latest(self, symbols: List[str]) -> Dict[str, float]:
        if not symbols:
            return {}
        with self.conn.cursor() as cur:
            cur.execute(
                """
                WITH ranked AS (
                    SELECT
                        symbol,
                        imbalance::float AS imbalance,
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn
                    FROM orderbook_l2
                    WHERE symbol = ANY(%s)
                )
                SELECT symbol, imbalance
                FROM ranked
                WHERE rn = 1
                """,
                ([s.upper() for s in symbols],),
            )
            return {str(r["symbol"]).upper(): float(r.get("imbalance") or 0.0) for r in cur.fetchall()}

    def get_funding_latest(self, symbols: List[str]) -> Dict[str, float]:
        if not symbols:
            return {}
        with self.conn.cursor() as cur:
            cur.execute(
                """
                WITH ranked AS (
                    SELECT
                        symbol,
                        funding_rate::float AS funding_rate,
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn
                    FROM funding_rates
                    WHERE symbol = ANY(%s)
                )
                SELECT symbol, funding_rate
                FROM ranked
                WHERE rn = 1
                """,
                ([s.upper() for s in symbols],),
            )
            return {str(r["symbol"]).upper(): float(r.get("funding_rate") or 0.0) for r in cur.fetchall()}

    def get_onchain_latest(self, symbols: List[str]) -> Dict[str, float]:
        if not symbols:
            return {}
        with self.conn.cursor() as cur:
            cur.execute(
                """
                WITH ranked AS (
                    SELECT
                        asset_symbol,
                        metric_value::float AS metric_value,
                        ROW_NUMBER() OVER (PARTITION BY asset_symbol ORDER BY ts DESC) AS rn
                    FROM onchain_signals
                    WHERE asset_symbol = ANY(%s)
                      AND metric_name IN ('netflow','exchange_netflow','net_inflow')
                )
                SELECT asset_symbol, metric_value
                FROM ranked
                WHERE rn = 1
                """,
                ([s.upper() for s in symbols],),
            )
            return {str(r["asset_symbol"]).upper(): float(r.get("metric_value") or 0.0) for r in cur.fetchall()}

    def get_event_context(self, target: str, limit: int = 5) -> List[Dict]:
        with self.conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT e.id, e.event_type, e.title, e.source_url, e.confidence_score, e.source_tier,
                           e.occurred_at, e.available_at, e.market_scope
                    FROM events e
                    LEFT JOIN event_links el ON el.event_id = e.id
                    LEFT JOIN entities en ON en.id = el.entity_id
                    WHERE en.symbol = UPPER(%s) OR en.name = %s
                    ORDER BY COALESCE(e.available_at, e.occurred_at) DESC
                    LIMIT %s
                    """,
                    (target, target, limit),
                )
            except Exception:
                cur.execute(
                    """
                    SELECT e.id, e.event_type, e.title, e.source_url, e.confidence_score, e.source_tier, e.occurred_at
                    FROM events e
                    LEFT JOIN event_links el ON el.event_id = e.id
                    LEFT JOIN entities en ON en.id = el.entity_id
                    WHERE en.symbol = UPPER(%s) OR en.name = %s
                    ORDER BY e.occurred_at DESC
                    LIMIT %s
                    """,
                    (target, target, limit),
                )
            return [dict(r) for r in cur.fetchall()]

    def get_event_contexts(self, targets: List[str], limit_per_target: int = 5) -> Dict[str, List[Dict]]:
        if not targets:
            return {}
        upper_targets = [t.upper() for t in targets]
        with self.conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    WITH linked AS (
                        SELECT
                            COALESCE(UPPER(en.symbol), en.name) AS target_key,
                            e.id, e.event_type, e.title, e.source_url, e.confidence_score, e.source_tier,
                            e.occurred_at, e.available_at, e.market_scope,
                            ROW_NUMBER() OVER (
                                PARTITION BY COALESCE(UPPER(en.symbol), en.name)
                                ORDER BY COALESCE(e.available_at, e.occurred_at) DESC
                            ) AS rn
                        FROM events e
                        LEFT JOIN event_links el ON el.event_id = e.id
                        LEFT JOIN entities en ON en.id = el.entity_id
                        WHERE UPPER(en.symbol) = ANY(%s) OR en.name = ANY(%s)
                    )
                    SELECT target_key, id, event_type, title, source_url, confidence_score, source_tier, occurred_at, available_at, market_scope
                    FROM linked
                    WHERE rn <= %s
                    ORDER BY target_key, COALESCE(available_at, occurred_at) DESC
                    """,
                    (upper_targets, targets, limit_per_target),
                )
                rows = [dict(r) for r in cur.fetchall()]
            except Exception:
                cur.execute(
                    """
                    WITH linked AS (
                        SELECT
                            COALESCE(UPPER(en.symbol), en.name) AS target_key,
                            e.id, e.event_type, e.title, e.source_url, e.confidence_score, e.source_tier, e.occurred_at,
                            ROW_NUMBER() OVER (
                                PARTITION BY COALESCE(UPPER(en.symbol), en.name)
                                ORDER BY e.occurred_at DESC
                            ) AS rn
                        FROM events e
                        LEFT JOIN event_links el ON el.event_id = e.id
                        LEFT JOIN entities en ON en.id = el.entity_id
                        WHERE UPPER(en.symbol) = ANY(%s) OR en.name = ANY(%s)
                    )
                    SELECT target_key, id, event_type, title, source_url, confidence_score, source_tier, occurred_at
                    FROM linked
                    WHERE rn <= %s
                    ORDER BY target_key, occurred_at DESC
                    """,
                    (upper_targets, targets, limit_per_target),
                )
                rows = [dict(r) for r in cur.fetchall()]
        grouped: Dict[str, List[Dict]] = {t: [] for t in upper_targets}
        for r in rows:
            key = str(r.pop("target_key")).upper()
            grouped.setdefault(key, []).append(r)
        # Fan-out global macro/policy events to all liquid targets so the model sees
        # centralized shocks even when no direct symbol entity link exists.
        with self.conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT id, event_type, title, source_url, confidence_score, source_tier, occurred_at, available_at, market_scope
                    FROM events
                    WHERE market_scope = 'macro'
                       OR COALESCE(payload->>'global_impact', 'false') = 'true'
                    ORDER BY COALESCE(available_at, occurred_at) DESC
                    LIMIT %s
                    """,
                    (max(1, limit_per_target),),
                )
                global_rows = [dict(r) for r in cur.fetchall()]
            except Exception:
                global_rows = []
        if global_rows:
            for tgt in upper_targets:
                seen = {int(x.get("id") or 0) for x in grouped.get(tgt, []) if x.get("id") is not None}
                merged = list(grouped.get(tgt, []))
                for gr in global_rows:
                    gid = int(gr.get("id") or 0)
                    if gid in seen:
                        continue
                    merged.append(dict(gr))
                merged.sort(
                    key=lambda x: (
                        x.get("available_at") or x.get("occurred_at") or datetime.fromtimestamp(0, tz=timezone.utc),
                    ),
                    reverse=True,
                )
                grouped[tgt] = merged[: max(1, limit_per_target)]
        return grouped

    def get_active_model(self, track: str) -> Dict[str, str]:
        now = time.time()
        cached = self.active_model_cache.get(track)
        if cached and now - float(cached["loaded_at"]) <= self.active_model_ttl_sec:
            return {"name": str(cached["name"]), "version": str(cached["version"])}

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT active_model_name, active_model_version
                FROM active_model_state
                WHERE track = %s
                """,
                (track,),
            )
            row = cur.fetchone()
            if not row:
                result = {
                    "name": "liquid_ttm_ensemble" if track == "liquid" else "vc_survival_ttm",
                    "version": "v2.1",
                }
            else:
                result = {"name": row["active_model_name"], "version": row["active_model_version"]}

            self.active_model_cache[track] = {
                "name": result["name"],
                "version": result["version"],
                "loaded_at": now,
            }
            return result

    def save_feature_snapshot(
        self,
        track: str,
        target: str,
        features: Dict[str, float],
        lineage_id: str,
        event_time: Optional[datetime] = None,
    ) -> None:
        ts = datetime.now(timezone.utc)
        with self.conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO feature_snapshots (
                        target, track, as_of, as_of_ts, event_time, feature_available_at, feature_version,
                        feature_payload, data_version, lineage_id, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        target,
                        track,
                        ts,
                        ts,
                        event_time or ts,
                        ts,
                        self.feature_version,
                        json.dumps(features),
                        self.data_version,
                        lineage_id,
                    ),
                )
            except Exception:
                cur.execute(
                    """
                    INSERT INTO feature_snapshots (
                        target, track, as_of, as_of_ts, event_time, feature_version,
                        feature_payload, data_version, lineage_id, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        target,
                        track,
                        ts,
                        ts,
                        event_time or ts,
                        self.feature_version,
                        json.dumps(features),
                        self.data_version,
                        lineage_id,
                    ),
                )

    def save_prediction_v2(
        self,
        track: str,
        target: str,
        score: float,
        confidence: float,
        outputs: Dict,
        explanation: Dict,
        lineage_id: str,
    ):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions_v2 (track, target, score, confidence, outputs, feature_set_id, decision_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING id
                """,
                (track, target, score, confidence, json.dumps(outputs), self.feature_version, lineage_id),
            )
            prediction_id = cur.fetchone()["id"]
            cur.execute(
                """
                INSERT INTO prediction_explanations (
                    prediction_id, top_event_contributors, top_feature_contributors,
                    evidence_links, model_version, feature_version, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """,
                (
                    prediction_id,
                    json.dumps(explanation.get("top_event_contributors", [])),
                    json.dumps(explanation.get("top_feature_contributors", [])),
                    json.dumps(explanation.get("evidence_links", [])),
                    explanation.get("model_version", "v2"),
                    explanation.get("feature_version", "feature-store-v2.0"),
                ),
            )
        self.conn.commit()
        return prediction_id

    def publish_signal(self, track: str, target: str, outputs: Dict):
        if not self.redis_client:
            return
        self.redis_client.xadd(
            "signal_stream",
            {
                "track": track,
                "target": target,
                "outputs": json.dumps(outputs),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def run_vc_once(self):
        targets = ["OpenAI", "Anthropic", "Scale AI"]
        active_model = self.get_active_model("vc")
        for name in targets:
            event_ctx = self.get_event_context(name)
            source_profile = self._event_quality_profile(event_ctx)
            feature = np.array([len(event_ctx), sum(float(e.get("confidence_score", 0.5)) for e in event_ctx), 1.0, 0.5, 0.2], dtype=np.float32)
            infer_lineage_id = f"infer-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{name.lower().replace(' ', '-')[:12]}"
            self.save_feature_snapshot(
                track="vc",
                target=name,
                features={
                    "event_count": float(feature[0]),
                    "event_conf_sum": float(feature[1]),
                    "bias_1": float(feature[2]),
                    "bias_2": float(feature[3]),
                    "bias_3": float(feature[4]),
                },
                lineage_id=infer_lineage_id,
                event_time=max((e.get("occurred_at") for e in event_ctx if e.get("occurred_at") is not None), default=None),
            )
            out = self.router.predict_vc(feature, model_name=active_model["name"])
            confidence = 0.55 + min(0.35, len(event_ctx) * 0.05)

            feature_contribs = [
                {"feature": "event_count", "value": len(event_ctx), "contribution": round(len(event_ctx) * 0.1, 4)},
                {
                    "feature": "mean_event_confidence",
                    "value": round(float(np.mean([e.get("confidence_score", 0.5) for e in event_ctx])) if event_ctx else 0.5, 4),
                    "contribution": 0.05,
                },
            ]
            exp = build_explanation(
                f"{active_model['name']}-{active_model['version']}",
                self.feature_version,
                event_ctx,
                feature_contribs,
                source_tier_summary=source_profile.get("source_tiers") or {},
                missing_flags=source_profile.get("missing_markers") or [],
            )
            score = float(out["p_next_round_12m"])
            out["lineage_id"] = infer_lineage_id
            out["data_version"] = self.data_version
            prediction_id = self.save_prediction_v2("vc", name, score, float(confidence), out, exp, lineage_id=infer_lineage_id)
            self.publish_signal("vc", name, {"prediction_id": prediction_id, **out})

    def run_liquid_once(self):
        active_model = self.get_active_model("liquid")
        latest_map = self.get_latest_prices(LIQUID_SYMBOLS)
        event_map = self.get_event_contexts(LIQUID_SYMBOLS, limit_per_target=8)
        orderbook_map = self.get_orderbook_imbalance_latest(LIQUID_SYMBOLS)
        funding_map = self.get_funding_latest(LIQUID_SYMBOLS)
        onchain_map = self.get_onchain_latest(LIQUID_SYMBOLS)
        for symbol in LIQUID_SYMBOLS:
            row = latest_map.get(symbol)
            if not row:
                continue
            event_ctx = event_map.get(symbol, [])
            source_profile = self._event_quality_profile(event_ctx)
            price = float(row["price"])
            volume = float(row.get("volume") or 0.0)
            history = self.get_recent_prices(symbol, limit=144)
            if len(history) >= 97:
                prices = np.array([float(h.get("price") or price) for h in history], dtype=np.float64)
                vols = np.array([float(h.get("volume") or 0.0) for h in history], dtype=np.float64)
                ret_1 = float((prices[-1] - prices[-2]) / max(prices[-2], 1e-12))
                ret_3 = float((prices[-1] - prices[-4]) / max(prices[-4], 1e-12))
                ret_12 = float((prices[-1] - prices[-13]) / max(prices[-13], 1e-12))
                ret_48 = float((prices[-1] - prices[-49]) / max(prices[-49], 1e-12))
                vol_3 = float(np.std(np.diff(np.log(np.clip(prices[-4:], 1e-12, None)))))
                vol_12 = float(np.std(np.diff(np.log(np.clip(prices[-13:], 1e-12, None)))))
                vol_48 = float(np.std(np.diff(np.log(np.clip(prices[-49:], 1e-12, None)))))
                vol_96 = float(np.std(np.diff(np.log(np.clip(prices[-97:], 1e-12, None)))))
                vol_hist = vols[-13:-1]
                vol_z = float((vols[-1] - np.mean(vol_hist)) / max(np.std(vol_hist), 1e-6))
                volume_impact = float(abs(ret_1) / max(np.sqrt(max(vols[-1], 1.0)), 1e-6))
            else:
                ret_1 = ret_3 = ret_12 = ret_48 = 0.0
                vol_3 = vol_12 = vol_48 = vol_96 = 0.0
                vol_z = 0.0
                volume_impact = 0.0

            orderbook_imbalance = float(orderbook_map.get(symbol, 0.0))
            funding_rate = float(funding_map.get(symbol, 0.0))
            onchain_flow = float(onchain_map.get(symbol, 0.0))
            onchain_norm = float(np.tanh(onchain_flow / 1e6))
            orderbook_missing_flag = 0.0 if symbol in orderbook_map else 1.0
            funding_missing_flag = 0.0 if symbol in funding_map else 1.0
            onchain_missing_flag = 0.0 if symbol in onchain_map else 1.0

            event_decay = float(source_profile.get("event_decay") or 0.0)

            feature = np.array(
                [
                    ret_1,
                    ret_3,
                    ret_12,
                    ret_48,
                    vol_3,
                    vol_12,
                    vol_48,
                    vol_96,
                    np.log1p(max(volume, 0.0)),
                    vol_z,
                    volume_impact,
                    orderbook_imbalance,
                    funding_rate,
                    onchain_norm,
                    event_decay,
                    orderbook_missing_flag,
                    funding_missing_flag,
                    onchain_missing_flag,
                ],
                dtype=np.float32,
            )
            infer_lineage_id = f"infer-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{symbol.lower()}"
            latest_event_time = None
            if event_ctx:
                latest_event_time = max((e.get("occurred_at") for e in event_ctx if e.get("occurred_at") is not None), default=None)
            self.save_feature_snapshot(
                track="liquid",
                target=symbol,
                features={
                    "ret_1": float(feature[0]),
                    "ret_3": float(feature[1]),
                    "ret_12": float(feature[2]),
                    "ret_48": float(feature[3]),
                    "vol_3": float(feature[4]),
                    "vol_12": float(feature[5]),
                    "vol_48": float(feature[6]),
                    "vol_96": float(feature[7]),
                    "log_volume": float(feature[8]),
                    "vol_z": float(feature[9]),
                    "volume_impact": float(feature[10]),
                    "orderbook_imbalance": float(feature[11]),
                    "funding_rate": float(feature[12]),
                    "onchain_norm": float(feature[13]),
                    "event_decay": float(feature[14]),
                    "orderbook_missing_flag": float(feature[15]),
                    "funding_missing_flag": float(feature[16]),
                    "onchain_missing_flag": float(feature[17]),
                    "source_tier_weight": float(source_profile.get("source_tier_weight") or 0.0),
                    "source_confidence": float(source_profile.get("source_confidence") or 0.0),
                    "feature_payload_schema_version": "v2.1",
                },
                lineage_id=infer_lineage_id,
                event_time=latest_event_time,
            )
            out = self.router.predict_liquid(symbol, feature, model_name=active_model["name"])

            feature_contribs = [
                {"feature": "ret_12", "value": round(float(feature[2]), 6), "contribution": round(float(feature[2]), 6)},
                {"feature": "volume_impact", "value": round(float(feature[10]), 6), "contribution": round(-float(feature[10]), 6)},
                {"feature": "orderbook_imbalance", "value": round(float(feature[11]), 6), "contribution": round(float(feature[11]) * 0.5, 6)},
            ]
            missing_markers = list(source_profile.get("missing_markers") or [])
            if orderbook_missing_flag > 0:
                missing_markers.append("orderbook_missing")
            if funding_missing_flag > 0:
                missing_markers.append("funding_missing")
            if onchain_missing_flag > 0:
                missing_markers.append("onchain_missing")
            exp = build_explanation(
                f"{active_model['name']}-{active_model['version']}",
                self.feature_version,
                event_ctx,
                feature_contribs,
                source_tier_summary=source_profile.get("source_tiers") or {},
                missing_flags=missing_markers,
            )
            score = float(out["expected_return"])
            output_payload = {**out, "symbol": symbol, "lineage_id": infer_lineage_id, "data_version": self.data_version}
            pred_id = self.save_prediction_v2("liquid", symbol, score, float(out["signal_confidence"]), output_payload, exp, lineage_id=infer_lineage_id)
            self.publish_signal("liquid", symbol, {"prediction_id": pred_id, **output_payload})

    async def run(self):
        self.connect()
        logger.info("inference-v2 started interval=%ss symbols=%s", INFER_INTERVAL_SEC, LIQUID_SYMBOLS)
        while True:
            try:
                self.run_vc_once()
                self.run_liquid_once()
                await asyncio.sleep(INFER_INTERVAL_SEC)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.error("inference-v2 cycle failed: %s", exc)
                await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(InferenceServiceV2().run())
