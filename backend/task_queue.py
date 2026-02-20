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
TASK_PROCESSING_KEY = os.getenv("TASK_PROCESSING_KEY", "ms:task_processing")
TASK_RESULT_PREFIX = os.getenv("TASK_RESULT_PREFIX", "ms:task_result:")
TASK_TTL_SECONDS = int(os.getenv("TASK_TTL_SECONDS", "86400"))
TASK_PROCESSING_TIMEOUT_SECONDS = int(os.getenv("TASK_PROCESSING_TIMEOUT_SECONDS", "900"))


def _client() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_task_body(raw: str) -> Optional[Dict[str, Any]]:
    try:
        body = json.loads(raw)
        return body if isinstance(body, dict) else None
    except Exception:
        return None


def _parse_iso(raw: Any) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _remove_from_processing_by_task_id(r: redis.Redis, task_id: str) -> None:
    if not task_id:
        return
    try:
        raws = r.lrange(TASK_PROCESSING_KEY, 0, -1)
    except Exception:
        return
    for raw in raws or []:
        body = _parse_task_body(raw)
        if not body:
            continue
        if str(body.get("task_id") or "") == task_id:
            try:
                r.lrem(TASK_PROCESSING_KEY, 0, raw)
            except Exception:
                pass


def _requeue_stale_processing_tasks(r: redis.Redis) -> None:
    timeout_s = max(1, int(TASK_PROCESSING_TIMEOUT_SECONDS))
    now = datetime.now(timezone.utc)
    try:
        raws = r.lrange(TASK_PROCESSING_KEY, 0, -1)
    except Exception:
        return
    for raw in raws or []:
        body = _parse_task_body(raw)
        if not body:
            try:
                r.lrem(TASK_PROCESSING_KEY, 1, raw)
            except Exception:
                pass
            continue
        claim_ts = _parse_iso(body.get("claimed_at")) or _parse_iso(body.get("created_at"))
        is_stale = claim_ts is None or (now - claim_ts).total_seconds() > float(timeout_s)
        if not is_stale:
            continue
        body["status"] = "queued"
        body["requeued_at"] = _now_iso()
        body["requeue_count"] = int(body.get("requeue_count") or 0) + 1
        raw_new = json.dumps(body)
        try:
            r.lrem(TASK_PROCESSING_KEY, 1, raw)
            r.lpush(TASK_QUEUE_KEY, raw_new)
        except Exception:
            continue


def enqueue_task(task_type: str, payload: Dict[str, Any]) -> str:
    task_id = f"task-{uuid.uuid4().hex}"
    body = {
        "task_id": task_id,
        "task_type": task_type,
        "payload": payload,
        "status": "queued",
        "created_at": _now_iso(),
        "claim_count": 0,
        "requeue_count": 0,
    }
    r = _client()
    r.setex(f"{TASK_RESULT_PREFIX}{task_id}", TASK_TTL_SECONDS, json.dumps(body))
    r.lpush(TASK_QUEUE_KEY, json.dumps(body))
    return task_id


def claim_next_task(timeout_seconds: int = 5) -> Optional[Dict[str, Any]]:
    r = _client()
    _requeue_stale_processing_tasks(r)
    raw: Optional[str] = None
    if hasattr(r, "brpoplpush"):
        raw = r.brpoplpush(TASK_QUEUE_KEY, TASK_PROCESSING_KEY, timeout=max(1, timeout_seconds))
    else:
        out = r.brpop(TASK_QUEUE_KEY, timeout=max(1, timeout_seconds))
        if out:
            _, raw = out
    if not raw:
        return None
    body = _parse_task_body(raw)
    if not body:
        try:
            r.lrem(TASK_PROCESSING_KEY, 1, raw)
        except Exception:
            pass
        return None
    body["claimed_at"] = _now_iso()
    body["claim_count"] = int(body.get("claim_count") or 0) + 1
    raw_new = json.dumps(body)
    if raw_new != raw:
        try:
            r.lrem(TASK_PROCESSING_KEY, 1, raw)
            r.lpush(TASK_PROCESSING_KEY, raw_new)
        except Exception:
            pass
    return body


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
    if status in {"completed", "failed"}:
        _remove_from_processing_by_task_id(r, task_id)


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
