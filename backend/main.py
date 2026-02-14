"""
FastAPI Main Application - Backend Service
"""
import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import List
import asyncio
import os

from gpu_manager import GPUManager
from nim_integration import get_nim_cache
from redis_streams import get_redis_consumer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
gpu_mgr = None
_nim_cache = None
_redis_consumer = None

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"✅ WebSocket connected (total: {len(self.active_connections)})")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"❌ WebSocket disconnected (total: {len(self.active_connections)})")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"❌ Failed to send WebSocket message: {e}")

manager = ConnectionManager()


async def consume_redis_messages():
    """Background task: Consume Redis Streams messages"""
    consumer = get_redis_consumer()
    logger.info("🔄 Starting Redis Streams consumer...")

    while True:
        try:
            # Consume news messages
            await asyncio.to_thread(consumer.consume, 'news_stream', count=1, block_ms=100)

            # Consume price messages
            await asyncio.to_thread(consumer.consume, 'price_stream', count=1, block_ms=100)

            # Consume prediction messages
            await asyncio.to_thread(consumer.consume, 'prediction_stream', count=1, block_ms=100)

            await asyncio.sleep(0.1)  # Prevent tight loop

        except Exception as e:
            logger.error(f"❌ Redis consumer error: {e}")
            await asyncio.sleep(1)  # Retry delay


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - Startup and shutdown"""
    global gpu_mgr, _nim_cache

    # Startup
    logger.info("🚀 Starting FastAPI backend...")

    # Initialize GPU manager
    gpu_mgr = GPUManager()
    system_status = gpu_mgr.get_status()
    logger.info(f"🎮 GPU Status: {system_status}")

    # Initialize NIM cache
    db_url = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")
    _nim_cache = get_nim_cache(db_url)
    logger.info("✅ NIM feature cache initialized")

    # Start background tasks
    asyncio.create_task(consume_redis_messages())

    yield

    # Shutdown
    logger.info("🛑 Shutting down FastAPI backend...")
    if _nim_cache:
        _nim_cache.close()


# Create FastAPI app
app = FastAPI(
    title="Monitoring System API",
    description="Backend API for Global Information Monitoring System",
    version="1.0.0",
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


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        gpu_status = gpu_mgr.get_status()
        return {
            "status": "healthy",
            "gpu": gpu_status,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        return {
            "system": gpu_mgr.get_status(),
            "services": {
                "backend": "running",
                "redis_consumer": "running"
            }
        }
    except Exception as e:
        logger.error(f"❌ Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/{symbol}")
async def get_predictions(symbol: str):
    """Get predictions for a symbol"""
    try:
        # Mock predictions (in production, inference service would provide this)
        return {
            "symbol": symbol,
            "predictions": [
                {
                    "horizon": "1h",
                    "direction": "up",
                    "change": "+1.2%",
                    "confidence": "high",
                    "score": 0.85,
                    "accuracy": 78
                },
                {
                    "horizon": "1d",
                    "direction": "up",
                    "change": "+3.5%",
                    "confidence": "medium",
                    "score": 0.73,
                    "accuracy": 73
                },
                {
                    "horizon": "7d",
                    "direction": "down",
                    "change": "-2.1%",
                    "confidence": "low",
                    "score": 0.61,
                    "accuracy": 61
                }
            ]
        }
    except Exception as e:
        logger.error(f"❌ Failed to get predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news")
async def get_news(symbol: str = None, limit: int = 20):
    """Get news items"""
    try:
        # Mock news data
        news = [
            {
                "id": 1,
                "title": "美联储暗示降息可能性增加",
                "time": "2026-02-14T12:45:00Z",
                "url": "https://example.com/news/1",
                "priority": "high",
                "sentiment": "positive",
                "is_important": True
            },
            {
                "id": 2,
                "title": "BTC 持续突破 $68,000",
                "time": "2026-02-14T12:30:00Z",
                "url": "https://example.com/news/2",
                "priority": "medium",
                "sentiment": "positive",
                "is_important": False
            },
            {
                "id": 3,
                "title": "欧盟通过 MiCA 加密货币监管法案",
                "time": "2026-02-14T12:15:00Z",
                "url": "https://example.com/news/3",
                "priority": "high",
                "sentiment": "negative",
                "is_important": True
            }
        ]

        if symbol:
            # Filter by symbol (mock)
            news = [item for item in news if symbol.upper() in item["title"]]

        return {"news": news[:limit], "count": len(news)}
    except Exception as e:
        logger.error(f"❌ Failed to get news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sentiment/{symbol}")
async def get_sentiment(symbol: str):
    """Get sentiment analysis for a symbol"""
    try:
        return {
            "symbol": symbol,
            "sentiment": {
                "positive": 65,
                "negative": 25,
                "neutral": 10
            },
            "overall": "positive",
            "trend": "up"
        }
    except Exception as e:
        logger.error(f"❌ Failed to get sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            # Process message
            if data.get("type") == "subscribe":
                symbol = data.get("symbol")
                logger.info(f"📡 Client subscribed to {symbol}")

                # Send welcome message
                await websocket.send_json({
                    "type": "subscribed",
                    "symbol": symbol,
                    "message": f"Subscribed to {symbol}"
                })

                # Start sending real-time data (mock)
                await send_realtime_updates(websocket, symbol)

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        manager.disconnect(websocket)


async def send_realtime_updates(websocket: WebSocket, symbol: str):
    """Send real-time price updates (mock)"""
    price = 67890.0

    for i in range(10):  # Send 10 updates
        try:
            price += 100 * (i % 2)
            change_percent = (price - 67890) / 67890 * 100

            await websocket.send_json({
                "type": "price_update",
                "symbol": symbol,
                "price": round(price, 2),
                "change": f"+{change_percent:.2f}%",
                "timestamp": asyncio.get_event_loop().time()
            })

            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"❌ Failed to send update: {e}")
            break


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
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
