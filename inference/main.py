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
from datetime import datetime
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
LIQUID_SYMBOLS = [s.strip().upper() for s in os.getenv("LIQUID_SYMBOLS", "BTC,ETH,AAPL,TSLA,NVDA,SPY").split(",") if s.strip()]
INFER_INTERVAL_SEC = int(os.getenv("INFER_INTERVAL_SEC", "60"))


class InferenceServiceV2:
    def __init__(self):
        self.router = ModelRouter()
        self.redis_client = None
        self.conn = None
        self.active_model_cache: Dict[str, Dict[str, object]] = {}
        self.active_model_ttl_sec = int(os.getenv("ACTIVE_MODEL_TTL_SEC", "30"))

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

    def get_event_context(self, target: str, limit: int = 5) -> List[Dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.id, e.event_type, e.title, e.source_url, e.confidence_score, e.occurred_at
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
            cur.execute(
                """
                WITH linked AS (
                    SELECT
                        COALESCE(UPPER(en.symbol), en.name) AS target_key,
                        e.id, e.event_type, e.title, e.source_url, e.confidence_score, e.occurred_at,
                        ROW_NUMBER() OVER (PARTITION BY COALESCE(UPPER(en.symbol), en.name) ORDER BY e.occurred_at DESC) AS rn
                    FROM events e
                    LEFT JOIN event_links el ON el.event_id = e.id
                    LEFT JOIN entities en ON en.id = el.entity_id
                    WHERE UPPER(en.symbol) = ANY(%s) OR en.name = ANY(%s)
                )
                SELECT target_key, id, event_type, title, source_url, confidence_score, occurred_at
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

    def save_prediction_v2(self, track: str, target: str, score: float, confidence: float, outputs: Dict, explanation: Dict):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions_v2 (track, target, score, confidence, outputs, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                RETURNING id
                """,
                (track, target, score, confidence, json.dumps(outputs)),
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
            feature = np.array([len(event_ctx), sum(float(e.get("confidence_score", 0.5)) for e in event_ctx), 1.0, 0.5, 0.2], dtype=np.float32)
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
            exp = build_explanation(f"{active_model['name']}-{active_model['version']}", "feature-store-v2.0", event_ctx, feature_contribs)
            score = float(out["p_next_round_12m"])
            prediction_id = self.save_prediction_v2("vc", name, score, float(confidence), out, exp)
            self.publish_signal("vc", name, {"prediction_id": prediction_id, **out})

    def run_liquid_once(self):
        active_model = self.get_active_model("liquid")
        latest_map = self.get_latest_prices(LIQUID_SYMBOLS)
        event_map = self.get_event_contexts(LIQUID_SYMBOLS, limit_per_target=8)
        for symbol in LIQUID_SYMBOLS:
            row = latest_map.get(symbol)
            if not row:
                continue
            event_ctx = event_map.get(symbol, [])
            price = float(row["price"])
            volume = float(row.get("volume") or 0.0)
            history = self.get_recent_prices(symbol, limit=64)
            if len(history) >= 20:
                prices = np.array([float(h.get("price") or price) for h in history], dtype=np.float64)
                vols = np.array([float(h.get("volume") or 0.0) for h in history], dtype=np.float64)
                ret_1 = float((prices[-1] - prices[-2]) / max(prices[-2], 1e-12))
                ret_3 = float((prices[-1] - prices[-4]) / max(prices[-4], 1e-12))
                ret_12 = float((prices[-1] - prices[-13]) / max(prices[-13], 1e-12))
                log_rets = np.diff(np.log(np.clip(prices[-13:], 1e-12, None)))
                vol_12 = float(np.std(log_rets))
                vol_hist = vols[-13:-1]
                vol_z = float((vols[-1] - np.mean(vol_hist)) / max(np.std(vol_hist), 1e-6))
            else:
                ret_1 = ret_3 = ret_12 = vol_12 = vol_z = 0.0
            event_decay = float(np.exp(-min(1.0, len(event_ctx) / 10.0)))
            feature = np.array(
                [ret_1, ret_3, ret_12, vol_12, np.log1p(max(volume, 0.0)), vol_z, 0.0, event_decay],
                dtype=np.float32,
            )
            out = self.router.predict_liquid(symbol, feature, model_name=active_model["name"])

            feature_contribs = [
                {"feature": "log_price", "value": round(float(feature[2]), 6), "contribution": 0.0},
                {"feature": "log_volume", "value": round(float(feature[1]), 6), "contribution": 0.0},
            ]
            exp = build_explanation(f"{active_model['name']}-{active_model['version']}", "feature-store-v2.0", event_ctx, feature_contribs)
            score = float(out["expected_return"])
            pred_id = self.save_prediction_v2("liquid", symbol, score, float(out["signal_confidence"]), {**out, "symbol": symbol}, exp)
            self.publish_signal("liquid", symbol, {"prediction_id": pred_id, **out, "symbol": symbol})

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
