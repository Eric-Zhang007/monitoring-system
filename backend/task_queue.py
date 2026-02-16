from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
TASK_QUEUE_KEY = os.getenv("TASK_QUEUE_KEY", "ms:task_queue")
TASK_RESULT_PREFIX = os.getenv("TASK_RESULT_PREFIX", "ms:task_result:")
TASK_TTL_SECONDS = int(os.getenv("TASK_TTL_SECONDS", "86400"))


def _client() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def enqueue_task(task_type: str, payload: Dict[str, Any]) -> str:
    task_id = f"task-{uuid.uuid4().hex}"
    body = {
        "task_id": task_id,
        "task_type": task_type,
        "payload": payload,
        "status": "queued",
        "created_at": _now_iso(),
    }
    r = _client()
    r.setex(f"{TASK_RESULT_PREFIX}{task_id}", TASK_TTL_SECONDS, json.dumps(body))
    r.lpush(TASK_QUEUE_KEY, json.dumps(body))
    return task_id


def claim_next_task(timeout_seconds: int = 5) -> Optional[Dict[str, Any]]:
    r = _client()
    out = r.brpop(TASK_QUEUE_KEY, timeout=max(1, timeout_seconds))
    if not out:
        return None
    _, raw = out
    try:
        body = json.loads(raw)
        if not isinstance(body, dict):
            return None
        return body
    except Exception:
        return None


def set_task_status(task_id: str, status: str, *, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
    key = f"{TASK_RESULT_PREFIX}{task_id}"
    r = _client()
    existing: Dict[str, Any] = {}
    raw = r.get(key)
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                existing = parsed
        except Exception:
            existing = {}
    existing["status"] = status
    existing["updated_at"] = _now_iso()
    if status == "running" and "started_at" not in existing:
        existing["started_at"] = existing["updated_at"]
    if status in {"completed", "failed"}:
        existing["finished_at"] = existing["updated_at"]
    if result is not None:
        existing["result"] = result
    if error is not None:
        existing["error"] = error
    r.setex(key, TASK_TTL_SECONDS, json.dumps(existing))


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    r = _client()
    raw = r.get(f"{TASK_RESULT_PREFIX}{task_id}")
    if not raw:
        return None
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def wait_task(task_id: str, timeout_seconds: int = 60) -> Optional[Dict[str, Any]]:
    deadline = time.monotonic() + max(1, timeout_seconds)
    while time.monotonic() < deadline:
        data = get_task(task_id)
        if data and data.get("status") in {"completed", "failed"}:
            return data
        time.sleep(0.5)
    return get_task(task_id)
