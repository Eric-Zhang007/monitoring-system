"""
FastAPI Main Application - Backend Service (修复版)
真实的后端API，不再使用Mock数据
"""
import os
import logging
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set
import asyncio
import json
from datetime import datetime, timedelta
from contextlib import suppress
from pathlib import Path

from gpu_manager import GPUManager
from nim_integration import get_nim_cache
from redis_streams import get_redis_consumer
from security_config import build_cors_settings
from v2_router import router as v2_router
from routers.multimodal import router as multimodal_router
from v2_repository import V2Repository
from metrics import (
    CONTENT_TYPE_LATEST,
    WEBSOCKET_ACTIVE_CONNECTIONS,
    WEBSOCKET_DROPPED_MESSAGES_TOTAL,
    observe_http_request,
    render_metrics,
    set_multimodal_health_metrics,
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
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
CORS_SETTINGS = build_cors_settings()
OPS_STATE_FILE = os.getenv(
    "OPS_STATE_FILE",
    str(Path(__file__).resolve().parents[1] / "artifacts" / "ops" / "continuous_runtime_state.json"),
)
HISTORY_COMPLETENESS_FILE = os.getenv(
    "HISTORY_COMPLETENESS_FILE",
    str(Path(__file__).resolve().parents[1] / "artifacts" / "audit" / "full_history_latest.json"),
)
ALIGNMENT_FILE = os.getenv(
    "ALIGNMENT_FILE",
    str(Path(__file__).resolve().parents[1] / "artifacts" / "audit" / "asof_alignment_latest.json"),
)
SOCIAL_THROUGHPUT_FILE = os.getenv(
    "SOCIAL_THROUGHPUT_FILE",
    str(Path(__file__).resolve().parents[1] / "artifacts" / "social" / "social_throughput_latest.json"),
)
MODEL_STATUS_FILE = os.getenv(
    "MODEL_STATUS_FILE",
    str(Path(__file__).resolve().parents[1] / "artifacts" / "models" / "candidate_registry.jsonl"),
)
PHASE_D_UNIFIED_SUMMARY_FILE = os.getenv(
    "PHASE_D_UNIFIED_SUMMARY_FILE",
    str(Path(__file__).resolve().parents[1] / "artifacts" / "experiments" / "phase_d_summary_latest.json"),
)
PAPER_STATE_FILE = os.getenv(
    "PAPER_STATE_FILE",
    str(Path(__file__).resolve().parents[1] / "artifacts" / "paper" / "paper_state.json"),
)
LIVE_CONTROL_FILE = os.getenv(
    "LIVE_CONTROL_FILE",
    str(Path(__file__).resolve().parents[1] / "artifacts" / "ops" / "live_control_state.json"),
)
MANUAL_CANDIDATE_FILE = os.getenv(
    "MANUAL_CANDIDATE_FILE",
    str(Path(__file__).resolve().parents[1] / "artifacts" / "ops" / "manual_candidate_control.json"),
)

# Global instances
gpu_mgr = None
_nim_cache = None
_redis_consumer = None
_postgres_conn = None
_redis_client = None
_v2_repo = None

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


def _get_v2_repo() -> V2Repository:
    global _v2_repo
    if _v2_repo is None:
        _v2_repo = V2Repository(DATABASE_URL)
    return _v2_repo


def _read_control_state_db(control_key: str) -> Optional[Dict[str, Any]]:
    try:
        row = _get_v2_repo().get_ops_control_state(control_key)
        if not row:
            return None
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        out = dict(payload)
        out.setdefault("status", "ok")
        out.setdefault("source", "db")
        out.setdefault("control_key", str(row.get("control_key") or str(control_key).strip().lower()))
        out.setdefault("updated_by", str(row.get("updated_by") or "system"))
        updated_at = row.get("updated_at")
        if updated_at:
            out.setdefault("updated_at", updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at))
        return out
    except Exception:
        return None


def _write_control_state_db(control_key: str, payload: Dict[str, Any], *, updated_by: str = "manual") -> Optional[Dict[str, Any]]:
    try:
        row = _get_v2_repo().upsert_ops_control_state(
            control_key=control_key,
            payload=payload,
            source="api",
            updated_by=updated_by or "manual",
        )
    except Exception:
        return None
    data = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    out = dict(data)
    out.setdefault("status", "ok")
    out.setdefault("source", "db")
    out.setdefault("control_key", str(row.get("control_key") or str(control_key).strip().lower()))
    out.setdefault("updated_by", str(row.get("updated_by") or "system"))
    updated_at = row.get("updated_at")
    if updated_at:
        out.setdefault("updated_at", updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at))
    return out


def _split_table_name(table_fqn: str) -> tuple[str, str]:
    if "." in table_fqn:
        schema, table = table_fqn.split(".", 1)
        return schema, table
    return "public", table_fqn


def _table_exists(cursor, table_fqn: str) -> bool:
    cursor.execute("SELECT to_regclass(%s) AS regclass", (table_fqn,))
    row = cursor.fetchone() or {}
    regclass = row.get("regclass") if isinstance(row, dict) else row[0]
    return regclass is not None


def _table_columns(cursor, schema: str, table: str) -> Set[str]:
    cursor.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        """,
        (schema, table),
    )
    rows = cursor.fetchall() or []
    out: Set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            val = row.get("column_name")
        else:
            val = row[0] if row else None
        if val:
            out.add(str(val))
    return out


def _safe_recent_count(
    cursor,
    *,
    table_fqn: str,
    time_columns: tuple[str, ...],
    lookback_hours: int,
) -> tuple[Optional[int], Optional[str]]:
    schema, table = _split_table_name(table_fqn)
    try:
        if not _table_exists(cursor, table_fqn):
            return None, f"optional probe skipped: {table_fqn} table missing"
        columns = _table_columns(cursor, schema, table)
    except Exception as exc:
        return None, f"optional probe skipped: {table_fqn} metadata lookup failed ({exc})"
    time_col = next((c for c in time_columns if c in columns), None)
    if not time_col:
        return None, (
            f"optional probe skipped: {table_fqn} missing time column "
            f"(expected one of {','.join(time_columns)})"
        )
    query = (
        f'SELECT COUNT(*) AS count FROM "{schema}"."{table}" '
        f'WHERE "{time_col}" > NOW() - make_interval(hours => %s)'
    )
    try:
        cursor.execute(query, (max(1, int(lookback_hours)),))
        row = cursor.fetchone() or {}
        value = row.get("count") if isinstance(row, dict) else row[0]
        return int(value or 0), None
    except Exception as exc:
        return None, f"optional probe failed: {table_fqn} ({exc})"


def _safe_recent_distinct_count(
    cursor,
    *,
    table_fqn: str,
    distinct_columns: tuple[str, ...],
    time_columns: tuple[str, ...],
    lookback_hours: int,
) -> tuple[Optional[int], Optional[str]]:
    schema, table = _split_table_name(table_fqn)
    try:
        if not _table_exists(cursor, table_fqn):
            return None, f"optional probe skipped: {table_fqn} table missing"
        columns = _table_columns(cursor, schema, table)
    except Exception as exc:
        return None, f"optional probe skipped: {table_fqn} metadata lookup failed ({exc})"
    time_col = next((c for c in time_columns if c in columns), None)
    distinct_col = next((c for c in distinct_columns if c in columns), None)
    if not time_col or not distinct_col:
        return None, (
            f"optional probe skipped: {table_fqn} missing required columns "
            f"(time={','.join(time_columns)} distinct={','.join(distinct_columns)})"
        )
    query = (
        f'SELECT COUNT(DISTINCT "{distinct_col}") AS count FROM "{schema}"."{table}" '
        f'WHERE "{time_col}" > NOW() - make_interval(hours => %s)'
    )
    try:
        cursor.execute(query, (max(1, int(lookback_hours)),))
        row = cursor.fetchone() or {}
        value = row.get("count") if isinstance(row, dict) else row[0]
        return int(value or 0), None
    except Exception as exc:
        return None, f"optional probe failed: {table_fqn} ({exc})"


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
    allow_origins=CORS_SETTINGS["allow_origins"],
    allow_credentials=CORS_SETTINGS["allow_credentials"],
    allow_methods=CORS_SETTINGS["allow_methods"],
    allow_headers=CORS_SETTINGS["allow_headers"],
)
app.include_router(v2_router)
app.include_router(multimodal_router)


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

        warnings: List[str] = []

        # 获取数据库中的符号数量（market_bars 优先，兼容历史 prices）
        conn = get_postgres()
        cursor = conn.cursor()
        active_symbols, active_symbols_warning = _safe_recent_distinct_count(
            cursor,
            table_fqn="public.market_bars",
            distinct_columns=("symbol", "target"),
            time_columns=("timestamp", "created_at", "ts"),
            lookback_hours=24,
        )
        if active_symbols is None:
            active_symbols, prices_warning = _safe_recent_distinct_count(
                cursor,
                table_fqn="public.prices",
                distinct_columns=("symbol", "target"),
                time_columns=("timestamp", "created_at", "ts"),
                lookback_hours=24,
            )
            if prices_warning:
                warnings.append(prices_warning)
        if active_symbols_warning:
            warnings.append(active_symbols_warning)

        recent_news, news_warning = _safe_recent_count(
            cursor,
            table_fqn="public.events",
            time_columns=("created_at", "timestamp", "occurred_at"),
            lookback_hours=24,
        )
        if news_warning:
            warnings.append(news_warning)

        recent_predictions, pred_warning = _safe_recent_count(
            cursor,
            table_fqn="public.predictions_v2",
            time_columns=("created_at", "timestamp", "ts"),
            lookback_hours=1,
        )
        if pred_warning:
            warnings.append(pred_warning)

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
            },
            "warnings": warnings,
        }
    except Exception as e:
        logger.error(f"❌ Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ops/runtime-state")
async def get_ops_runtime_state():
    """Return latest continuous paper+train runtime state for frontend monitoring."""
    try:
        db_state = _read_control_state_db("ops_runtime_state")
        if db_state:
            return db_state
        p = Path(OPS_STATE_FILE)
        if not p.exists():
            return {"status": "missing", "path": str(p)}
        raw = p.read_text(encoding="utf-8")
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            raise RuntimeError("invalid_ops_state_not_object")
        payload.setdefault("status", "ok")
        payload.setdefault("path", str(p))
        return payload
    except Exception as e:
        logger.error(f"❌ Failed to read ops runtime state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _read_json_file(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        return {"status": "missing", "path": str(p)}
    raw = p.read_text(encoding="utf-8")
    obj = json.loads(raw) if raw.strip() else {}
    if isinstance(obj, dict):
        obj.setdefault("status", "ok")
        obj.setdefault("path", str(p))
        return obj
    return {"status": "invalid", "path": str(p)}


def _read_last_jsonl(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        return {"status": "missing", "path": str(p)}
    last = ""
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line.strip()
    if not last:
        return {"status": "missing", "path": str(p)}
    try:
        obj = json.loads(last)
        if isinstance(obj, dict):
            obj.setdefault("status", "ok")
            obj.setdefault("path", str(p))
            return obj
    except Exception:
        pass
    return {"status": "invalid", "path": str(p)}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_multimodal_health(
    *,
    latest_candidate: Dict[str, Any],
    phase_d_summary: Dict[str, Any],
    social_throughput: Dict[str, Any],
) -> Dict[str, Any]:
    cand = latest_candidate if isinstance(latest_candidate, dict) else {}
    summary = phase_d_summary if isinstance(phase_d_summary, dict) else {}
    social = social_throughput if isinstance(social_throughput, dict) else {}
    gate = cand.get("gate") if isinstance(cand.get("gate"), dict) else {}
    ab = cand.get("ablation_summary") if isinstance(cand.get("ablation_summary"), dict) else {}
    backbones = list(cand.get("backbone_ready_list") or [])
    if not backbones:
        bb = summary.get("backbone") if isinstance(summary.get("backbone"), dict) else {}
        backbones = list(bb.get("ready_backbones") or [])

    gate_openness = _to_float(cand.get("gate_val_mean"), 0.0)
    text_contribution_abs = _to_float(cand.get("delta_val_mean_abs"), 0.0)
    if text_contribution_abs <= 0.0:
        text_contribution_abs = max(
            0.0,
            _to_float(ab.get("delta_mse_no_text_vs_full"), 0.0),
        )

    text_coverage_ratio = _to_float(social.get("text_coverage_ratio"), -1.0)
    if text_coverage_ratio < 0:
        rows = ab.get("rows") if isinstance(ab.get("rows"), dict) else {}
        full_rows = _to_float(rows.get("full"), 0.0)
        event_rows = _to_float(rows.get("event_window"), 0.0)
        if full_rows > 0:
            text_coverage_ratio = max(0.0, min(1.0, event_rows / full_rows))
        else:
            text_coverage_ratio = 0.0

    candidate_gate_passed = bool(gate.get("passed", False))
    health = {
        "status": "ok",
        "fusion_mode": str(cand.get("fusion_mode") or ""),
        "text_dropout_prob": _to_float(cand.get("text_dropout_prob"), 0.0),
        "text_coverage_ratio": float(max(0.0, min(1.0, text_coverage_ratio))),
        "gate_openness": float(max(0.0, min(1.0, gate_openness))),
        "text_contribution_abs": float(max(0.0, text_contribution_abs)),
        "ready_backbones": [str(x) for x in backbones if str(x).strip()],
        "ready_backbone_count": int(len([x for x in backbones if str(x).strip()])),
        "candidate_gate_passed": bool(candidate_gate_passed),
        "candidate_gate_reasons": list(gate.get("reasons") or []),
        "ablation_primary": str(ab.get("primary") or summary.get("ablation", {}).get("primary_ablation") or ""),
    }
    return health


@app.get("/api/v2/monitor/history-completeness")
async def monitor_history_completeness():
    try:
        return _read_json_file(HISTORY_COMPLETENESS_FILE)
    except Exception as e:
        logger.error(f"❌ Failed to read history completeness snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/monitor/alignment")
async def monitor_alignment():
    try:
        return _read_json_file(ALIGNMENT_FILE)
    except Exception as e:
        logger.error(f"❌ Failed to read alignment snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/monitor/social-throughput")
async def monitor_social_throughput():
    try:
        db_state = _read_control_state_db("social_throughput")
        if db_state:
            return db_state
        return _read_json_file(SOCIAL_THROUGHPUT_FILE)
    except Exception as e:
        logger.error(f"❌ Failed to read social throughput snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/monitor/model-status")
async def monitor_model_status():
    try:
        latest_candidate = _read_last_jsonl(MODEL_STATUS_FILE)
        db_candidate = _read_control_state_db("manual_candidate")
        if db_candidate:
            latest_candidate = db_candidate
        phase_d_summary = _read_json_file(PHASE_D_UNIFIED_SUMMARY_FILE)
        social_state = _read_control_state_db("social_throughput") or _read_json_file(SOCIAL_THROUGHPUT_FILE)
        multimodal_health = _extract_multimodal_health(
            latest_candidate=latest_candidate,
            phase_d_summary=phase_d_summary,
            social_throughput=social_state if isinstance(social_state, dict) else {},
        )
        set_multimodal_health_metrics(
            track="liquid",
            text_coverage_ratio=float(multimodal_health.get("text_coverage_ratio", 0.0)),
            gate_openness=float(multimodal_health.get("gate_openness", 0.0)),
            text_contribution_abs=float(multimodal_health.get("text_contribution_abs", 0.0)),
            ready_backbones=int(multimodal_health.get("ready_backbone_count", 0)),
            candidate_gate_passed=bool(multimodal_health.get("candidate_gate_passed", False)),
        )
        active = {}
        try:
            conn = get_postgres()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT track, active_model_name, active_model_version
                FROM active_model_state
                ORDER BY track ASC
                """
            )
            active = {str(r.get("track")): {"name": r.get("active_model_name"), "version": r.get("active_model_version")} for r in (cur.fetchall() or [])}
            cur.close()
            conn.close()
        except Exception:
            active = {}
        return {
            "status": "ok",
            "active_models": active,
            "latest_candidate": latest_candidate,
            "multimodal_health": multimodal_health,
            "phase_d_summary": phase_d_summary,
        }
    except Exception as e:
        logger.error(f"❌ Failed to read model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/monitor/multimodal-health")
async def monitor_multimodal_health():
    try:
        latest_candidate = _read_last_jsonl(MODEL_STATUS_FILE)
        phase_d_summary = _read_json_file(PHASE_D_UNIFIED_SUMMARY_FILE)
        social_state = _read_control_state_db("social_throughput") or _read_json_file(SOCIAL_THROUGHPUT_FILE)
        health = _extract_multimodal_health(
            latest_candidate=latest_candidate,
            phase_d_summary=phase_d_summary,
            social_throughput=social_state if isinstance(social_state, dict) else {},
        )
        set_multimodal_health_metrics(
            track="liquid",
            text_coverage_ratio=float(health.get("text_coverage_ratio", 0.0)),
            gate_openness=float(health.get("gate_openness", 0.0)),
            text_contribution_abs=float(health.get("text_contribution_abs", 0.0)),
            ready_backbones=int(health.get("ready_backbone_count", 0)),
            candidate_gate_passed=bool(health.get("candidate_gate_passed", False)),
        )
        return health
    except Exception as e:
        logger.error(f"❌ Failed to read multimodal health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/monitor/paper-performance")
async def monitor_paper_performance():
    try:
        db_state = _read_control_state_db("paper_performance")
        if db_state:
            return db_state
        return _read_json_file(PAPER_STATE_FILE)
    except Exception as e:
        logger.error(f"❌ Failed to read paper performance snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/control/live/state")
async def control_live_state():
    try:
        db_state = _read_control_state_db("live_control_state")
        if db_state:
            return db_state
        return _read_json_file(LIVE_CONTROL_FILE)
    except Exception as e:
        logger.error(f"❌ Failed to read live control state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/control/live/enable")
async def control_live_enable(payload: Dict = Body(default={})):
    try:
        current = _read_control_state_db("live_control_state") or _read_json_file(LIVE_CONTROL_FILE)
        if not isinstance(current, dict):
            current = {}
        current["live_enabled"] = True
        current["paper_enabled"] = bool(payload.get("paper_enabled", False))
        current["updated_at"] = datetime.utcnow().isoformat() + "Z"
        current["operator"] = str(payload.get("operator") or "manual")
        current["reason"] = str(payload.get("reason") or "manual_enable")
        db_saved = _write_control_state_db(
            "live_control_state",
            current,
            updated_by=str(current.get("operator") or "manual"),
        )
        p = Path(LIVE_CONTROL_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(db_saved or {"status": "ok", **current})
    except Exception as e:
        logger.error(f"❌ Failed to enable live control: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/control/live/disable")
async def control_live_disable(payload: Dict = Body(default={})):
    try:
        current = _read_control_state_db("live_control_state") or _read_json_file(LIVE_CONTROL_FILE)
        if not isinstance(current, dict):
            current = {}
        current["live_enabled"] = False
        current["paper_enabled"] = bool(payload.get("paper_enabled", True))
        current["updated_at"] = datetime.utcnow().isoformat() + "Z"
        current["operator"] = str(payload.get("operator") or "manual")
        current["reason"] = str(payload.get("reason") or "manual_disable")
        db_saved = _write_control_state_db(
            "live_control_state",
            current,
            updated_by=str(current.get("operator") or "manual"),
        )
        p = Path(LIVE_CONTROL_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(db_saved or {"status": "ok", **current})
    except Exception as e:
        logger.error(f"❌ Failed to disable live control: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/control/model/candidate-state")
async def control_candidate_state():
    try:
        db_state = _read_control_state_db("manual_candidate")
        if db_state:
            return db_state
        return _read_json_file(MANUAL_CANDIDATE_FILE)
    except Exception as e:
        logger.error(f"❌ Failed to read candidate control state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/control/model/switch-candidate")
async def control_switch_candidate(payload: Dict = Body(default={})):
    try:
        row = {
            "status": "ok",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "track": str(payload.get("track") or "liquid"),
            "model_name": str(payload.get("model_name") or ""),
            "model_version": str(payload.get("model_version") or ""),
            "operator": str(payload.get("operator") or "manual"),
            "reason": str(payload.get("reason") or "manual_candidate_switch"),
        }
        db_saved = _write_control_state_db(
            "manual_candidate",
            row,
            updated_by=str(row.get("operator") or "manual"),
        )
        p = Path(MANUAL_CANDIDATE_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(db_saved or row)
    except Exception as e:
        logger.error(f"❌ Failed to switch candidate: {e}")
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


@app.get("/api/predictions/latest")
async def get_latest_predictions():
    _legacy_api_frozen()


# ======================================
# Price Endpoints (修复版：真实数据)
# ======================================

@app.get("/api/prices/{symbol}")
async def get_prices(
    symbol: str,
    hours: int = Query(default=24, ge=1, le=168, description="过去多少小时的价格")
):
    _legacy_api_frozen()


@app.get("/api/prices/{symbol}/latest")
async def get_latest_price(symbol: str):
    _legacy_api_frozen()


# ======================================
# News Endpoints (修复版：真实数据)
# ======================================

@app.get("/api/news")
async def get_news(
    symbol: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100, description="返回数量")
):
    _legacy_api_frozen()


@app.get("/api/news/{symbol}/sentiment")
async def get_symbol_sentiment(
    symbol: str,
    hours: int = Query(default=24, ge=1, le=168)
):
    _legacy_api_frozen()


# ======================================
# Technical Indicators Endpoints
# ======================================

@app.get("/api/indicators/{symbol}")
async def get_technical_indicators(symbol: str):
    _legacy_api_frozen()


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
