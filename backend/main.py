"""
FastAPI Main Application - Backend Service (修复版)
真实的后端API，不再使用Mock数据
"""
import os
import logging
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Set
import asyncio
import json
from datetime import datetime, timedelta
from contextlib import suppress

from gpu_manager import GPUManager
from nim_integration import get_nim_cache
from redis_streams import get_redis_consumer
from v2_router import router as v2_router
from metrics import (
    CONTENT_TYPE_LATEST,
    WEBSOCKET_ACTIVE_CONNECTIONS,
    WEBSOCKET_DROPPED_MESSAGES_TOTAL,
    observe_http_request,
    render_metrics,
)
import psycopg2
from psycopg2.extras import RealDictCursor
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

# Global instances
gpu_mgr = None
_nim_cache = None
_redis_consumer = None
_postgres_conn = None
_redis_client = None

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, Set[str]] = {}  # websocket -> set of symbols
        self.channels: Dict[WebSocket, str] = {}  # websocket -> stream channel
        self.queues: Dict[WebSocket, asyncio.Queue] = {}
        self.sender_tasks: Dict[WebSocket, asyncio.Task] = {}
        self.queue_max = int(os.getenv("WS_QUEUE_MAX", "256"))
        self.batch_max = int(os.getenv("WS_BATCH_MAX", "32"))
        self.flush_ms = int(os.getenv("WS_FLUSH_MS", "120"))
        self.send_timeout_sec = float(os.getenv("WS_SEND_TIMEOUT_SEC", "1.0"))

    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = set()
        self.channels[websocket] = channel
        q: asyncio.Queue = asyncio.Queue(maxsize=max(1, self.queue_max))
        self.queues[websocket] = q
        self.sender_tasks[websocket] = asyncio.create_task(self._sender_loop(websocket, q, channel))
        WEBSOCKET_ACTIVE_CONNECTIONS.set(len(self.active_connections))
        logger.info(f"✅ WebSocket connected (total: {len(self.active_connections)})")

    def disconnect(self, websocket: WebSocket, reason: str = "disconnect"):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.subscriptions.pop(websocket, None)
        self.channels.pop(websocket, None)
        self.queues.pop(websocket, None)
        task = self.sender_tasks.pop(websocket, None)
        if task:
            task.cancel()
        with suppress(Exception):
            asyncio.create_task(websocket.close(code=1000))
        WEBSOCKET_ACTIVE_CONNECTIONS.set(len(self.active_connections))
        logger.info(f"❌ WebSocket disconnected (reason={reason}, total: {len(self.active_connections)})")

    async def _sender_loop(self, websocket: WebSocket, q: asyncio.Queue, channel: str):
        flush_sec = max(0.01, self.flush_ms / 1000.0)
        try:
            while True:
                first = await q.get()
                batch = [first]
                deadline = time.perf_counter() + flush_sec
                while len(batch) < self.batch_max:
                    remain = deadline - time.perf_counter()
                    if remain <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(q.get(), timeout=remain)
                        batch.append(nxt)
                    except asyncio.TimeoutError:
                        break
                payload = batch[0] if len(batch) == 1 else {
                    "type": "batch",
                    "channel": channel,
                    "count": len(batch),
                    "timestamp": datetime.utcnow().isoformat(),
                    "items": batch,
                }
                await asyncio.wait_for(websocket.send_json(payload), timeout=self.send_timeout_sec)
        except asyncio.CancelledError:
            return
        except Exception as e:
            WEBSOCKET_DROPPED_MESSAGES_TOTAL.labels(reason="send_error").inc()
            logger.error(f"❌ WebSocket sender failed: {e}")
            self.disconnect(websocket, reason="send_error")

    def _enqueue(self, websocket: WebSocket, message: dict):
        q = self.queues.get(websocket)
        if not q:
            return
        if q.full():
            WEBSOCKET_DROPPED_MESSAGES_TOTAL.labels(reason="queue_full").inc()
            self.disconnect(websocket, reason="queue_full")
            return
        with suppress(Exception):
            q.put_nowait(message)

    def subscribe(self, websocket: WebSocket, symbol: str):
        """订阅符号"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].add(symbol)
            logger.info(f"📡 WebSocket subscribed to {symbol}")

    def unsubscribe(self, websocket: WebSocket, symbol: str):
        """取消订阅符号"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].discard(symbol)

    async def broadcast_symbol(self, symbol: str, message: dict):
        """广播消息给订阅该符号的所有连接"""
        for connection in list(self.active_connections):
            channel = self.channels.get(connection, "")
            if channel == "signals" and symbol in self.subscriptions.get(connection, set()):
                self._enqueue(connection, message)

    async def broadcast(self, message: dict):
        """广播给所有连接（系统消息）"""
        for connection in list(self.active_connections):
            self._enqueue(connection, message)

manager = ConnectionManager()


def _legacy_api_frozen():
    raise HTTPException(
        status_code=410,
        detail="Legacy v1 endpoints are frozen. Use /api/v2/* endpoints only.",
    )


# 数据库连接函数
def get_postgres():
    """获取PostgreSQL连接"""
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")


def get_redis():
    """获取Redis连接"""
    try:
        client = redis.from_url(REDIS_URL, decode_responses=True)
        return client
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        raise HTTPException(status_code=500, detail="Redis connection failed")


async def consume_redis_messages():
    """后台任务：消费Redis Streams消息"""
    consumer = get_redis_consumer()
    logger.info("🔄 Starting Redis Streams consumer...")

    while True:
        try:
            # 消费预测消息
            await asyncio.to_thread(consumer.consume, 'prediction_stream', count=10, block_ms=100)

            # 消费价格消息
            await asyncio.to_thread(consumer.consume, 'price_stream', count=10, block_ms=100)

            # 消费新闻消息
            await asyncio.to_thread(consumer.consume, 'news_stream', count=10, block_ms=100)

            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"❌ Redis consumer error: {e}")
            await asyncio.sleep(1)


async def broadcast_predictions():
    """后台任务：从Redis读取预测并广播给WebSocket客户端"""
    r = get_redis()
    stream_name = "prediction_stream"

    while True:
        try:
            # 读取最新的预测消息
            messages = r.xrevrange(stream_name, count=10)

            for msg_id, msg_data in messages:
                try:
                    symbol = msg_data.get('symbol')
                    if symbol:
                        # 广播给订阅该符号的连接
                        await manager.broadcast_symbol(symbol, {
                            "type": "prediction",
                            "data": msg_data,
                            "timestamp": msg_data.get("timestamp")
                        })
                except Exception as e:
                    logger.error(f"❌ Failed to broadcast prediction: {e}")

            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"❌ Broadcast error: {e}")
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global gpu_mgr, _nim_cache, _postgres_conn, _redis_client

    # 启动
    logger.info("🚀 Starting FastAPI backend...")

    # 初始化GPU管理器
    gpu_mgr = GPUManager()
    system_status = gpu_mgr.get_status()
    logger.info(f"🎮 GPU Status: {system_status}")

    # 初始化NIM缓存
    _nim_cache = get_nim_cache(DATABASE_URL)
    logger.info("✅ NIM feature cache initialized")

    # 连接数据库
    try:
        _postgres_conn = get_postgres()
        _redis_client = get_redis()
        logger.info("✅ Connected to databases")
    except Exception as e:
        logger.warning(f"⚠️ Database connection failed (Will retry): {e}")

    # 启动后台任务
    asyncio.create_task(consume_redis_messages())
    asyncio.create_task(broadcast_predictions())

    yield

    # 关闭
    logger.info("🛑 Shutting down FastAPI backend...")
    if _nim_cache:
        _nim_cache.close()
    if _postgres_conn:
        _postgres_conn.close()


# 创建FastAPI应用
app = FastAPI(
    title="Monitoring System API",
    description="Backend API for Global Information Monitoring System",
    version="1.0.1",  # 修复版
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(v2_router)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        raise
    finally:
        elapsed = time.perf_counter() - start
        observe_http_request(request.method, request.url.path, status_code, elapsed)


@app.get("/metrics")
async def prometheus_metrics():
    return PlainTextResponse(render_metrics(), media_type=CONTENT_TYPE_LATEST)


# ======================================
# Health & Status Endpoints
# ======================================

@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        gpu_status = gpu_mgr.get_status() if gpu_mgr else {}
        redis_status = "connected" if _redis_client else "disconnected"
        postgres_status = "connected" if _postgres_conn else "disconnected"

        return {
            "status": "healthy",
            "gpu": gpu_status,
            "redis": redis_status,
            "postgres": postgres_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_system_status():
    """获取系统状态"""
    try:
        gpu_status = gpu_mgr.get_status() if gpu_mgr else {}

        # 获取数据库中的符号数量
        conn = get_postgres()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM prices WHERE timestamp > NOW() - INTERVAL '24 hours'")
        active_symbols = cursor.fetchone()['count'] or 0

        cursor.execute("SELECT COUNT(*) FROM events WHERE created_at > NOW() - INTERVAL '24 hours'")
        recent_news = cursor.fetchone()['count'] or 0

        cursor.execute("SELECT COUNT(*) FROM predictions_v2 WHERE created_at > NOW() - INTERVAL '1 hour'")
        recent_predictions = cursor.fetchone()['count'] or 0

        cursor.close()
        conn.close()

        return {
            "system": {
                "gpu": gpu_status,
                "redis": "connected" if _redis_client else "disconnected",
                "postgres": "connected" if _postgres_conn else "disconnected"
            },
            "data": {
                "active_symbols": active_symbols,
                "recent_news_24h": recent_news,
                "recent_predictions_1h": recent_predictions
            },
            "services": {
                "backend": "running",
                "redis_consumer": "running",
                "websocket_broadcaster": "running"
            }
        }
    except Exception as e:
        logger.error(f"❌ Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================
# Prediction Endpoints (修复版：真实数据)
# ======================================

@app.get("/api/predictions/{symbol}")
async def get_predictions(
    symbol: str,
    hours: int = Query(default=24, ge=1, le=168, description="过去多少小时的预测")
):
    _legacy_api_frozen()
    """
    获取预测数据（修复版：从数据库查询真实预测）

    Args:
        symbol: 交易对/股票符号
        hours: 查询过去几小时的预测
    """
    try:
        conn = get_postgres()
        cursor = conn.cursor()

        query = """
            SELECT
                symbol,
                scenario,
                direction,
                confidence,
                expected_change_pct,
                expected_price,
                scenario_probabilities,
                created_at
            FROM predictions
            WHERE symbol = UPPER(%s)
              AND created_at > NOW() - make_interval(hours => %s)
            ORDER BY created_at DESC
            LIMIT 100
        """

        cursor.execute(query, (symbol, hours))
        rows = cursor.fetchall()

        predictions = []
        for row in rows:
            pred = dict(row)
            pred['scenario_probabilities'] = json.loads(pred['scenario_probabilities'])
            predictions.append(pred)

        cursor.close()
        conn.close()

        return {
            "symbol": symbol.upper(),
            "predictions": predictions,
            "count": len(predictions)
        }

    except Exception as e:
        logger.error(f"❌ Failed to get predictions for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/latest")
async def get_latest_predictions():
    _legacy_api_frozen()
    """获取所有符号的最新预测"""
    try:
        conn = get_postgres()
        cursor = conn.cursor()

        # 使用窗口函数获取每个符号的最新预测
        query = """
            WITH ranked_predictions AS (
                SELECT
                    symbol,
                    scenario,
                    direction,
                    confidence,
                    expected_change_pct,
                    expected_price,
                    scenario_probabilities,
                    created_at,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY created_at DESC) as rn
                FROM predictions
                WHERE created_at > NOW() - INTERVAL '24 hours'
            )
            SELECT * FROM ranked_predictions WHERE rn = 1
            ORDER BY created_at DESC
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        predictions = []
        for row in rows:
            pred = dict(row)
            pred.pop('rn')
            pred['scenario_probabilities'] = json.loads(pred['scenario_probabilities'])
            predictions.append(pred)

        cursor.close()
        conn.close()

        return {
            "predictions": predictions,
            "count": len(predictions)
        }

    except Exception as e:
        logger.error(f"❌ Failed to get latest predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================
# Price Endpoints (修复版：真实数据)
# ======================================

@app.get("/api/prices/{symbol}")
async def get_prices(
    symbol: str,
    hours: int = Query(default=24, ge=1, le=168, description="过去多少小时的价格")
):
    _legacy_api_frozen()
    """
    获取价格历史（修复版：从数据库查询真实价格）

    Args:
        symbol: 交易对/股票符号
        hours: 查询过去几小时的价格
    """
    try:
        # 先尝试Redis缓存
        r = get_redis()
        cache_key = f"prices:{symbol}"
        cached = r.get(cache_key)

        if cached:
            return json.loads(cached)

        # Redis没有，查询数据库
        conn = get_postgres()
        cursor = conn.cursor()

        query = """
            SELECT
                symbol,
                price,
                volume,
                timestamp
            FROM prices
            WHERE symbol = UPPER(%s)
              AND timestamp > NOW() - make_interval(hours => %s)
            ORDER BY timestamp DESC
            LIMIT 500
        """

        cursor.execute(query, (symbol, hours))
        rows = cursor.fetchall()

        prices = [dict(row) for row in rows]

        # 缓存到Redis（5分钟）
        r.setex(cache_key, 300, json.dumps({"symbol": symbol.upper(), "prices": prices, "count": len(prices)}))

        cursor.close()
        conn.close()

        return {
            "symbol": symbol.upper(),
            "prices": prices,
            "count": len(prices)
        }

    except Exception as e:
        logger.error(f"❌ Failed to get prices for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prices/{symbol}/latest")
async def get_latest_price(symbol: str):
    _legacy_api_frozen()
    """获取最新价格"""
    try:
        r = get_redis()
        cache_key = f"price:{symbol}"
        cached = r.get(cache_key)

        if cached:
            return json.loads(cached)

        conn = get_postgres()
        cursor = conn.cursor()

        query = """
            SELECT price, volume, timestamp
            FROM prices
            WHERE symbol = UPPER(%s)
            ORDER BY timestamp DESC
            LIMIT 1
        """

        cursor.execute(query, (symbol,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"No price data found for {symbol}")

        result = dict(row)

        # 缓存到Redis（1分钟）
        r.setex(cache_key, 60, json.dumps(result))

        cursor.close()
        conn.close()

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get latest price for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================
# News Endpoints (修复版：真实数据)
# ======================================

@app.get("/api/news")
async def get_news(
    symbol: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100, description="返回数量")
):
    _legacy_api_frozen()
    """
    获取新闻（修复版：从数据库查询真实新闻）

    Args:
        symbol: 可选，过滤符号
        limit: 返回数量
    """
    try:
        conn = get_postgres()
        cursor = conn.cursor()

        if symbol:
            query = """
                SELECT
                    id,
                    title,
                    url,
                    symbol,
                    priority,
                    sentiment,
                    is_important,
                    summary,
                    created_at
                FROM news
                WHERE symbol = UPPER(%s)
                ORDER BY created_at DESC
                LIMIT %s
            """
            cursor.execute(query, (symbol, limit))
        else:
            query = """
                SELECT
                    id,
                    title,
                    url,
                    symbol,
                    priority,
                    sentiment,
                    is_important,
                    summary,
                    created_at
                FROM news
                WHERE is_important = TRUE
                ORDER BY created_at DESC
                LIMIT %s
            """
            cursor.execute(query, (limit,))

        rows = cursor.fetchall()
        news_items = [dict(row) for row in rows]

        cursor.close()
        conn.close()

        return {
            "news": news_items,
            "count": len(news_items)
        }

    except Exception as e:
        logger.error(f"❌ Failed to get news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/{symbol}/sentiment")
async def get_symbol_sentiment(
    symbol: str,
    hours: int = Query(default=24, ge=1, le=168)
):
    _legacy_api_frozen()
    """
    获取符号的情感分析（修复版：从数据库查询真实情感）

    Args:
        symbol: 交易对/股票符号
        hours: 过去多少小时
    """
    try:
        conn = get_postgres()
        cursor = conn.cursor()

        query = """
            SELECT
                sentiment,
                COUNT(*) as count
            FROM news
            WHERE symbol = UPPER(%s)
              AND created_at > NOW() - make_interval(hours => %s)
            GROUP BY sentiment
        """

        cursor.execute(query, (symbol, hours))
        rows = cursor.fetchall()

        sentiment_counts = {row['sentiment']: row['count'] for row in rows}
        total = sum(sentiment_counts.values()) or 1

        # 计算百分比
        sentiment_percentages = {
            k: round(v / total * 100, 1) for k, v in sentiment_counts.items()
        }

        # 确定总体情感
        positive = sentiment_percentages.get('positive', 0)
        negative = sentiment_percentages.get('negative', 0)

        if positive > negative + 20:
            overall = "positive"
            trend = "up"
        elif negative > positive + 20:
            overall = "negative"
            trend = "down"
        else:
            overall = "neutral"
            trend = "sideways"

        cursor.close()
        conn.close()

        return {
            "symbol": symbol.upper(),
            "sentiment": {
                "counts": sentiment_counts,
                "percentages": sentiment_percentages
            },
            "overall": overall,
            "trend": trend,
            "total_news": total if total > 1 else 0,
            "timeframe": f"{hours}h"
        }

    except Exception as e:
        logger.error(f"❌ Failed to get sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================
# Technical Indicators Endpoints
# ======================================

@app.get("/api/indicators/{symbol}")
async def get_technical_indicators(symbol: str):
    _legacy_api_frozen()
    """获取技术指标"""
    try:
        conn = get_postgres()
        cursor = conn.cursor()

        query = """
            SELECT * FROM v_latest_ti
            WHERE symbol = UPPER(%s)
            LIMIT 1
        """

        cursor.execute(query, (symbol,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"No technical indicators found for {symbol}")

        cursor.close()
        conn.close()

        return dict(row)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get indicators for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================
# WebSocket Endpoint (修复版：真实数据推送)
# ======================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json(
        {
            "type": "error",
            "status": 410,
            "detail": "Legacy /ws endpoint is frozen. Use /stream/events|signals|risk.",
        }
    )
    await websocket.close(code=1008)
    return


@app.websocket("/stream/events")
async def websocket_events(websocket: WebSocket):
    """V2 events stream channel."""
    await manager.connect(websocket, channel="events")
    await websocket.send_json({"type": "events", "status": "subscribed", "timestamp": datetime.utcnow().isoformat()})
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, reason="client_disconnect")


@app.websocket("/stream/signals")
async def websocket_signals(websocket: WebSocket):
    """V2 signals stream channel."""
    await manager.connect(websocket, channel="signals")
    await websocket.send_json({"type": "signals", "status": "subscribed", "timestamp": datetime.utcnow().isoformat()})
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw) if raw else {}
            except Exception:
                msg = {}
            msg_type = str(msg.get("type") or "").lower()
            symbol = str(msg.get("symbol") or "").upper()
            if msg_type == "subscribe" and symbol:
                manager.subscribe(websocket, symbol)
                await websocket.send_json({"type": "signals", "status": "subscribed_symbol", "symbol": symbol, "timestamp": datetime.utcnow().isoformat()})
            elif msg_type == "unsubscribe" and symbol:
                manager.unsubscribe(websocket, symbol)
                await websocket.send_json({"type": "signals", "status": "unsubscribed_symbol", "symbol": symbol, "timestamp": datetime.utcnow().isoformat()})
            else:
                await websocket.send_json({"type": "signals", "status": "noop", "timestamp": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        manager.disconnect(websocket, reason="client_disconnect")


@app.websocket("/stream/risk")
async def websocket_risk(websocket: WebSocket):
    """V2 risk stream channel."""
    await manager.connect(websocket, channel="risk")
    await websocket.send_json({"type": "risk", "status": "subscribed", "timestamp": datetime.utcnow().isoformat()})
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, reason="client_disconnect")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        manager.disconnect(websocket, reason="server_error")


# ======================================
# Global Exception Handler
# ======================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"❌ Unhandled exception: {exc}")
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
