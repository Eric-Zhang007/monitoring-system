"""
Redis Streams Producer and Consumer (Alternative to Kafka)
- Fixed v3: Added message acknowledgment mechanism
- XACK for message confirmation
- Consumer groups for reliable delivery
"""
import redis
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

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

    def consume(self, stream_name: str, count: int = 10, block_ms: int = 1000):
        """
        Consume messages (Fixed v3: Added XACK message acknowledgment)
        """
        try:
            # 1. Create consumer group if not exists
            try:
                self.redis.xgroup_create(
                    stream_name,
                    self.consumer_group,
                    id='0',
                    mkstream=True
                )
                logger.info(f"✅ Created consumer group: {self.consumer_group}")
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    logger.error(f"❌ Failed to create consumer group: {e}")

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
                # TODO: Store to database, trigger NIM extraction, etc.

            elif stream == 'price_stream':
                price = data.get('price', 'N/A')
                symbol = data.get('symbol', 'N/A')
                logger.info(f"💰 Processing price: {symbol} = {price}")
                # TODO: Store to ClickHouse, trigger predictions

            elif stream == 'prediction_stream':
                horizon = data.get('horizon', 'N/A')
                symbol = data.get('symbol', 'N/A')
                logger.info(f"🎯 Processing prediction: {symbol} {horizon}")
                # TODO: Store to database, send via WebSocket

        except Exception as e:
            logger.error(f"❌ Processing logic failed: {e}")
            raise

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
        _redis_producer = RedisProducer(redis_url or "redis://localhost:6379")
    return _redis_producer

def get_redis_consumer(redis_url: str = None) -> RedisConsumer:
    """Get global Redis consumer instance"""
    global _redis_consumer
    if _redis_consumer is None:
        _redis_consumer = RedisConsumer(redis_url or "redis://localhost:6379")
    return _redis_consumer
