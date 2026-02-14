"""
Data Collector - Collects news, prices, and market data
Sends data to Redis Streams
"""
import redis
import json
import time
import logging
from datetime import datetime
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """Data collector for news, prices, and other market data"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.stream_names = {
            'news': 'news_stream',
            'price': 'price_stream'
        }

    def produce_news(self, news_data: dict):
        """Send news to Redis Streams"""
        try:
            message_id = self.redis.xadd(self.stream_names['news'], news_data)
            logger.info(f"📰 News sent: {message_id} - {news_data.get('title', 'N/A')}")
            return message_id
        except Exception as e:
            logger.error(f"❌ Failed to send news: {e}")
            return None

    def produce_price(self, price_data: dict):
        """Send price data to Redis Streams"""
        try:
            message_id = self.redis.xadd(self.stream_names['price'], price_data)
            logger.info(f"💹 Price sent: {message_id} - {price_data.get('symbol', 'N/A')} = {price_data.get('price', 'N/A')}")
            return message_id
        except Exception as e:
            logger.error(f"❌ Failed to send price: {e}")
            return None

    def collect_mock_news(self, count: int = 5):
        """Collect mock news data (for MVP testing)"""
        news_templates = [
            {
                "priority": "high",
                "sentiment": "positive",
                "title": "美联储暗示降息可能性增加"
            },
            {
                "priority": "high",
                "sentiment": "positive",
                "title": "BTC 持续突破 $68,000"
            },
            {
                "priority": "medium",
                "sentiment": "negative",
                "title": "某科技公司财报不及预期"
            },
            {
                "priority": "low",
                "sentiment": "neutral",
                "title": "市场成交量温和上涨"
            },
            {
                "priority": "high",
                "sentiment": "negative",
                "title": "欧盟通过新的加密货币监管法案"
            }
        ]

        symbols = ["BTC", "ETH", "AAPL", "TSLA", "SPY"]
        collected = []

        for i in range(count):
            template = random.choice(news_templates)
            news = {
                "title": template["title"],
                "symbol": random.choice(symbols),
                "priority": template["priority"],
                "sentiment": template["sentiment"],
                "time": datetime.utcnow().isoformat() + "Z",
                "url": f"https://example.com/news/{i}",
                "summary": template["title"]  # Using title as summary for MVP
            }

            self.produce_news(news)
            collected.append(news)

        return collected

    def collect_mock_price(self, symbols: list = None, count: int = 10):
        """Collect mock price data (for MVP testing)"""
        if not symbols:
            symbols = ["BTC", "ETH", "AAPL", "TSLA", "SPY"]

        base_prices = {
            "BTC": 67890.0,
            "ETH": 3450.0,
            "AAPL": 185.0,
            "TSLA": 235.0,
            "SPY": 478.0
        }

        collected = []

        for i in range(count):
            symbol = random.choice(symbols)
            base_price = base_prices[symbol]
            price = base_price + random.uniform(-500, 500) if symbol in ["BTC", "ETH"] else base_price + random.uniform(-10, 10)

            price_data = {
                "symbol": symbol,
                "price": round(price, 2),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "volume": random.randint(1000, 100000)
            }

            self.produce_price(price_data)
            collected.append(price_data)

        return collected

    def run(self, interval: int = 60):
        """Run collector continuously"""
        logger.info(f"🚀 Starting data collector (interval: {interval}s)")

        while True:
            try:
                logger.info("📥 Collecting data...")

                # Collect news
                news = self.collect_mock_news(count=5)
                logger.info(f"✅ Collected {len(news)} news items")

                # Collect prices
                prices = self.collect_mock_price(count=10)
                logger.info(f"✅ Collected {len(prices)} price items")

                logger.info(f"⏳ Waiting {interval} seconds...")
                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("🛑 Collector stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Collector error: {e}")
                time.sleep(10)  # Retry after 10 seconds

    def close(self):
        """Close Redis connection"""
        if self.redis:
            self.redis.close()
            logger.info("✅ Collector connection closed")


if __name__ == "__main__":
    collector = DataCollector()
    try:
        collector.run(interval=30)
    except KeyboardInterrupt:
        collector.close()
