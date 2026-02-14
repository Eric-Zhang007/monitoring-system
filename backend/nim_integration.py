"""
NIM Integration - Offline Feature Extraction + Online Cache (Fixed v3)
- NIM offline extraction on GPU 1 (every 6 hours)
- Online cache retrieval from PGVector (< 10ms)
- SQL Injection FIX: Using make_interval for parameterized queries
"""
import asyncio
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
import logging

logger = logging.getLogger(__name__)

class NIMFeatureCache:
    """
    NIM Offline Feature Cache (Fixed v3):
    - Offline: Extract semantic features every 6 hours (GPU 1)
    - Online: Read cached features from PGVector (GPU 0)
    - FIXED: SQL injection vulnerability with make_interval
    - Added: last_news_time tracking
    """

    def __init__(self, db_url: str):
        self.conn = psycopg2.connect(db_url)
        register_vector(self.conn)
        self._ensure_table()

    def _ensure_table(self):
        """Ensure PGVector table exists (FIXED: added last_news_time)"""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS semantic_features (
                    symbol VARCHAR(20) PRIMARY KEY,
                    feature_vector vector(384),
                    timestamp TIMESTAMP DEFAULT NOW(),
                    last_news_time TIMESTAMP,  -- ✅ NEW: Last news timestamp
                    news_count INTEGER
                )
            """)
            self.conn.commit()

    async def offline_extraction(self, symbol: str, news_items: list):
        """
        Offline feature extraction (every 6 hours, GPU 1)
        """
        if not news_items:
            logger.warning(f"⚠️  {symbol}: No news, skipping extraction")
            return

        # Filter low-quality news
        filtered_news = self._filter_low_quality(news_items)
        if not filtered_news:
            logger.warning(f"⚠️  {symbol}: No valid news after filtering")
            return

        # Build batch prompt
        prompt = self._build_batch_prompt(filtered_news)

        # Call NIM to generate feature vector (single API call)
        feature_vector = await self._call_nim_embedding(prompt)

        # Store in PGVector (FIXED: added last_news_time)
        with self.conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO semantic_features
                    (symbol, feature_vector, timestamp, last_news_time, news_count)
                    VALUES (%s, %s, NOW(), %s, %s)
                    ON CONFLICT (symbol) DO UPDATE SET
                        feature_vector = EXCLUDED.feature_vector,
                        timestamp = EXCLUDED.timestamp,
                        last_news_time = EXCLUDED.last_news_time,
                        news_count = EXCLUDED.news_count
                """, (
                    symbol,
                    feature_vector,
                    filtered_news[0].get('time'),
                    len(filtered_news)
                ))
                self.conn.commit()
                logger.info(f"✅ {symbol}: Extracted semantic features from {len(filtered_news)} news items")
            except Exception as e:
                logger.error(f"❌ {symbol}: Failed to store features: {e}")
                self.conn.rollback()

    def get_cached_features(self, symbol: str, max_age_hours: int = 6):
        """
        Online retrieval of cached features (real-time prediction, GPU 0)

        Returns: feature_vector or None (if expired)
        """
        with self.conn.cursor() as cur:
            try:
                # ✅ FIXED SQL Injection: Using make_interval with parameterization
                cur.execute("""
                    SELECT feature_vector, last_news_time, news_count
                    FROM semantic_features
                    WHERE symbol = %s
                    AND timestamp > NOW() - make_interval(secs => %s * 3600)
                """, (symbol, max_age_hours))

                result = cur.fetchone()
                if result:
                    feature_vector, last_news_time, news_count = result
                    logger.debug(f"📦 {symbol}: Retrieved cached features (age < {max_age_hours}h)")
                    return feature_vector
                logger.info(f"⏰ {symbol}: No cached features (expired or not found)")
                return None
            except Exception as e:
                logger.error(f"❌ {symbol}: Failed to retrieve cached features: {e}")
                return None

    def _filter_low_quality(self, news_items: list) -> list:
        """Filter low-quality news"""
        filtered = []
        for item in news_items:
            title = item.get('title', '')
            if len(title) < 10:
                continue
            if '广告' in title or '推广' in title:
                continue
            filtered.append(item)
        return filtered

    def _build_batch_prompt(self, news_items: list) -> str:
        """Build batch news analysis prompt"""
        texts = [f"{i+1}. {item['title']} - {item.get('summary', '')}"
                for i, item in enumerate(news_items[:20])]

        return f"""Analyze the comprehensive market impact of the following {len(texts)} financial news items:

{chr(10).join(texts)}

Extract key investment insights and output structured analysis results."""

    async def _call_nim_embedding(self, prompt: str) -> list:
        """
        Call NIM to generate embedding vector
        Cost estimate: ~¥0.01 per call
        """
        # TODO: Actual implementation with NVIDIA NIM API
        # from nvidia import nim_sdk
        # response = await nim_sdk.embed(prompt, model="nv-embed-v1")
        # return response.embedding

        # Mock implementation for MVP
        # In production, this would call NIM on GPU 1
        await asyncio.sleep(0.01)  # Simulate API call
        return np.random.randn(384).tolist()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("✅ NIM feature cache connection closed")

# Global instance
_nim_cache = None

def get_nim_cache(db_url: str = None):
    """Get global NIM cache instance"""
    global _nim_cache
    if _nim_cache is None and db_url:
        _nim_cache = NIMFeatureCache(db_url)
    return _nim_cache
