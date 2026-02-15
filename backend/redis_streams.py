"""
Redis Streams Producer and Consumer (Alternative to Kafka)
- Fixed v3: Added message acknowledgment mechanism
- XACK for message confirmation
- Consumer groups for reliable delivery
"""
import redis
import json
import logging
import os
from typing import Dict, Any, Optional
import hashlib
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")

class RedisProducer:
    """Redis Streams Producer"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.stream_names = {
            'news': 'news_stream',
            'price': 'price_stream',
            'prediction': 'prediction_stream'
        }

    def send_message(self, message_type: str, data: Dict[str, Any]) -> Optional[str]:
        """Send message to Redis Streams"""
        stream_name = self.stream_names.get(message_type)
        if not stream_name:
            logger.error(f"❌ Unknown message type: {message_type}")
            return None

        try:
            message_id = self.redis.xadd(stream_name, data)
            logger.info(f"📤 Sent message: {stream_name} -> {message_id}")
            return message_id
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            return None

    def close(self):
        """Close Redis connection"""
        if self.redis:
            self.redis.close()
            logger.info("✅ Redis producer connection closed")


class RedisConsumer:
    """Redis Streams Consumer (Fixed v3: Added XACK message acknowledgment)"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.consumer_group = "backend_group"
        self.consumer_name = f"consumer_{id(self)}"  # Unique consumer name
        self._initialized_streams = set()

    def _ensure_group(self, stream_name: str):
        if stream_name in self._initialized_streams:
            return
        try:
            self.redis.xgroup_create(
                stream_name,
                self.consumer_group,
                id='0',
                mkstream=True
            )
            logger.info(f"✅ Created consumer group: {self.consumer_group} stream={stream_name}")
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"❌ Failed to create consumer group: {e}")
        self._initialized_streams.add(stream_name)

    def consume(self, stream_name: str, count: int = 10, block_ms: int = 1000):
        """
        Consume messages (Fixed v3: Added XACK message acknowledgment)
        """
        try:
            # 1. Ensure consumer group once per stream.
            self._ensure_group(stream_name)

            # 2. Read messages
            messages = self.redis.xreadgroup(
                self.consumer_group,
                self.consumer_name,
                {stream_name: '>'},
                count=count,
                block=block_ms
            )

            if not messages:
                return []

            processed = []
            for stream, message_list in messages:
                for message_id, fields in message_list:
                    try:
                        # 3. Process message
                        data = fields  # Already decoded with decode_responses=True
                        logger.info(f"📥 Received message: {message_id}")

                        # Call business logic processing
                        self.process_message(stream, data)

                        # ✅ 4. Acknowledge message processing success
                        self.redis.xack(stream_name, self.consumer_group, message_id)
                        logger.debug(f"✅ Acknowledged message: {message_id}")

                        processed.append((stream, message_id, data))

                    except Exception as e:
                        logger.error(f"❌ Failed to process message {message_id}: {e}")
                        # Message not acknowledged, will be retried

            return processed

        except Exception as e:
            logger.error(f"❌ Consumer exception: {e}")
            return []

    def process_message(self, stream: str, data: Dict[str, str]):
        """Process message (business logic placeholder)"""
        try:
            if stream == 'news_stream':
                title = data.get('title', 'N/A')
                logger.info(f"📰 Processing news: {title}")
                self._store_news_event(data)

            elif stream == 'price_stream':
                price = data.get('price', 'N/A')
                symbol = data.get('symbol', 'N/A')
                logger.info(f"💰 Processing price: {symbol} = {price}")
                self._store_price_tick(data)

            elif stream == 'prediction_stream':
                horizon = data.get('horizon', 'N/A')
                symbol = data.get('symbol', 'N/A')
                logger.info(f"🎯 Processing prediction: {symbol} {horizon}")
                self._store_prediction_signal(data)

        except Exception as e:
            logger.error(f"❌ Processing logic failed: {e}")
            raise

    @staticmethod
    def _connect_pg():
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

    def _store_news_event(self, data: Dict[str, str]):
        title = data.get("title") or "Untitled"
        source_url = data.get("url") or ""
        symbol = (data.get("symbol") or "").upper()
        summary = data.get("summary") or ""
        occurred_at = data.get("time") or datetime.utcnow().isoformat() + "Z"
        fingerprint = hashlib.sha256(f"market|{title.lower()}|{source_url}|{occurred_at}".encode("utf-8")).hexdigest()
        with self._connect_pg() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO events (
                        event_type, title, occurred_at, source_url, source_name, source_timezone,
                        source_tier, confidence_score, event_importance, novelty_score,
                        entity_confidence, payload, fingerprint, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, 'UTC',
                        %s, %s, %s, %s,
                        %s, %s, %s, NOW()
                    )
                    ON CONFLICT (fingerprint) DO NOTHING
                    RETURNING id
                    """,
                    (
                        "market",
                        title,
                        occurred_at,
                        source_url,
                        "redis-news",
                        3,
                        0.55,
                        0.5,
                        0.5,
                        0.4,
                        json.dumps({"summary": summary[:1200]}),
                        fingerprint,
                    ),
                )
                row = cur.fetchone()
                if not row:
                    return
                event_id = row["id"]
                if symbol and symbol != "OTHER":
                    cur.execute(
                        """
                        INSERT INTO entities (entity_type, name, symbol, metadata, created_at, updated_at)
                        VALUES ('asset', %s, %s, '{}'::jsonb, NOW(), NOW())
                        ON CONFLICT (entity_type, name) DO UPDATE SET symbol = EXCLUDED.symbol, updated_at = NOW()
                        RETURNING id
                        """,
                        (symbol, symbol),
                    )
                    entity_id = cur.fetchone()["id"]
                    cur.execute(
                        """
                        INSERT INTO event_links (event_id, entity_id, role, created_at)
                        VALUES (%s, %s, 'mentioned', NOW())
                        ON CONFLICT (event_id, entity_id, role) DO NOTHING
                        """,
                        (event_id, entity_id),
                    )

    def _store_price_tick(self, data: Dict[str, str]):
        symbol = (data.get("symbol") or "").upper()
        if not symbol:
            return
        try:
            price = float(data.get("price") or 0.0)
        except Exception:
            return
        volume = int(float(data.get("volume") or 0.0))
        ts = data.get("timestamp") or datetime.utcnow().isoformat() + "Z"
        with self._connect_pg() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prices (symbol, price, volume, timestamp, created_at)
                    VALUES (UPPER(%s), %s, %s, %s, NOW())
                    """,
                    (symbol, price, volume, ts),
                )

    def _store_prediction_signal(self, data: Dict[str, str]):
        target = (data.get("symbol") or data.get("target") or "UNKNOWN").upper()
        track = data.get("track") or "liquid"
        horizon = data.get("horizon") or "1d"
        outputs = data.get("outputs")
        if isinstance(outputs, str):
            try:
                outputs = json.loads(outputs)
            except Exception:
                outputs = {}
        if not isinstance(outputs, dict):
            outputs = {}
        score = float(outputs.get("expected_return", 0.0) or 0.0)
        confidence = float(outputs.get("signal_confidence", 0.5) or 0.5)
        action = "buy" if score > 0.001 else "sell" if score < -0.001 else "hold"
        with self._connect_pg() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO signal_candidates (
                        track, target, horizon, score, confidence,
                        action, policy, decision_id, metadata, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        track,
                        target,
                        horizon,
                        score,
                        confidence,
                        action,
                        "redis-stream",
                        f"redis-{hashlib.md5((target+horizon).encode('utf-8')).hexdigest()[:12]}",
                        json.dumps({"source": "prediction_stream"}),
                    ),
                )

    def get_pending_messages(self, stream_name: str):
        """Get pending messages (for recovery)"""
        try:
            messages = self.redis.xpending_range(
                stream_name,
                self.consumer_group,
                min="-",
                max="+",
                count=100
            )
            return messages
        except Exception as e:
            logger.error(f"❌ Failed to get pending messages: {e}")
            return []

    def close(self):
        """Close Redis connection"""
        if self.redis:
            self.redis.close()
            logger.info("✅ Redis consumer connection closed")


# Singleton instances
_redis_producer = None
_redis_consumer = None

def get_redis_producer(redis_url: str = None) -> RedisProducer:
    """Get global Redis producer instance"""
    global _redis_producer
    if _redis_producer is None:
        _redis_producer = RedisProducer(redis_url or os.getenv("REDIS_URL", "redis://localhost:6379"))
    return _redis_producer

def get_redis_consumer(redis_url: str = None) -> RedisConsumer:
    """Get global Redis consumer instance"""
    global _redis_consumer
    if _redis_consumer is None:
        _redis_consumer = RedisConsumer(redis_url or os.getenv("REDIS_URL", "redis://localhost:6379"))
    return _redis_consumer
