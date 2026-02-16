#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

APP_ROOT = Path(__file__).resolve().parents[1]
IMPORT_PATHS = ["/app", str(APP_ROOT / "backend")]
for p in IMPORT_PATHS:
    if p not in sys.path:
        sys.path.append(p)

from task_queue import claim_next_task, set_task_status
from schemas_v2 import BacktestRunRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
POLL_TIMEOUT_SEC = int(os.getenv("TASK_WORKER_POLL_TIMEOUT_SEC", "5"))
WORKER_ERROR_BACKOFF_BASE_SEC = float(os.getenv("TASK_WORKER_ERROR_BACKOFF_BASE_SEC", "1.0"))
WORKER_ERROR_BACKOFF_MAX_SEC = float(os.getenv("TASK_WORKER_ERROR_BACKOFF_MAX_SEC", "30.0"))


def _run_pnl_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    from v2_repository import V2Repository

    track = str(payload.get("track") or "liquid")
    lookback_hours = int(payload.get("lookback_hours") or 24 * 7)
    repo = V2Repository(DATABASE_URL)
    attribution = repo.build_pnl_attribution(track=track, lookback_hours=lookback_hours)
    return {
        "track": track,
        "lookback_hours": lookback_hours,
        "totals": attribution.get("totals", {}),
        "by_target": attribution.get("by_target", []),
    }


async def _run_backtest_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    from v2_router import run_backtest

    req = BacktestRunRequest(**payload)
    resp = await run_backtest(req)
    return resp.model_dump()


class BacktestRuntime:
    """Persistent async runtime for async jobs; avoids per-task asyncio.run()."""

    def __init__(self) -> None:
        self._runner: Optional[asyncio.Runner] = None

    def run(self, coro) -> Dict[str, Any]:
        if self._runner is None:
            self._runner = asyncio.Runner()
        return self._runner.run(coro)

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None


def _execute_task(
    task_type: str,
    payload: Dict[str, Any],
    *,
    backtest_runtime: Optional[BacktestRuntime] = None,
) -> Dict[str, Any]:
    if task_type == "backtest_run":
        runtime = backtest_runtime or BacktestRuntime()
        try:
            return runtime.run(_run_backtest_job(payload))
        finally:
            if backtest_runtime is None:
                runtime.close()
    if task_type == "pnl_attribution":
        return _run_pnl_job(payload)
    raise ValueError(f"unsupported_task_type:{task_type}")


class TaskWorker:
    """Single worker loop with a persistent runtime for async task handlers."""

    def __init__(
        self,
        *,
        poll_timeout_sec: int = POLL_TIMEOUT_SEC,
        claim_task_fn: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
        set_status_fn: Optional[Callable[..., None]] = None,
        backtest_runtime: Optional[BacktestRuntime] = None,
    ) -> None:
        self.poll_timeout_sec = max(1, int(poll_timeout_sec))
        self._claim_task_fn = claim_task_fn or claim_next_task
        self._set_status_fn = set_status_fn or set_task_status
        self._backtest_runtime = backtest_runtime or BacktestRuntime()
        self._owns_runtime = backtest_runtime is None

    def _claim_task(self) -> Optional[Dict[str, Any]]:
        return self._claim_task_fn(timeout_seconds=self.poll_timeout_sec)

    def _set_status(
        self,
        task_id: str,
        status: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        self._set_status_fn(task_id, status, result=result, error=error)

    def _safe_set_status(
        self,
        task_id: str,
        status: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        try:
            self._set_status(task_id, status, result=result, error=error)
            return True
        except Exception as exc:
            logger.error("set task status failed id=%s status=%s err=%s", task_id, status, exc)
            return False

    def close(self) -> None:
        if self._owns_runtime:
            self._backtest_runtime.close()

    def run(self, *, max_tasks: int | None = None) -> None:
        logger.info("task worker started poll_timeout_sec=%s", self.poll_timeout_sec)
        handled = 0
        claim_backoff_sec = max(0.1, float(WORKER_ERROR_BACKOFF_BASE_SEC))
        try:
            while True:
                if max_tasks is not None and handled >= max_tasks:
                    return
                try:
                    task = self._claim_task()
                    claim_backoff_sec = max(0.1, float(WORKER_ERROR_BACKOFF_BASE_SEC))
                except Exception as exc:
                    logger.error("claim_next_task failed err=%s\n%s", exc, traceback.format_exc())
                    time.sleep(claim_backoff_sec)
                    claim_backoff_sec = min(
                        max(claim_backoff_sec * 2.0, claim_backoff_sec + 0.5),
                        max(0.5, float(WORKER_ERROR_BACKOFF_MAX_SEC)),
                    )
                    continue
                if not task:
                    continue
                task_id = str(task.get("task_id") or "")
                task_type = str(task.get("task_type") or "")
                payload_obj = task.get("payload") or {}
                payload = payload_obj if isinstance(payload_obj, dict) else {}
                if not task_id or not task_type:
                    logger.warning("skip malformed task payload=%s", task)
                    continue
                self._safe_set_status(task_id, "running")
                try:
                    result = _execute_task(task_type, payload, backtest_runtime=self._backtest_runtime)
                    self._safe_set_status(task_id, "completed", result=result)
                    logger.info("task completed id=%s type=%s", task_id, task_type)
                except Exception as exc:
                    self._safe_set_status(task_id, "failed", error=f"{exc}")
                    logger.error("task failed id=%s type=%s err=%s\n%s", task_id, task_type, exc, traceback.format_exc())
                handled += 1
        finally:
            self.close()


def run_worker(*, max_tasks: int | None = None) -> None:
    worker = TaskWorker()
    worker.run(max_tasks=max_tasks)


def main() -> None:
    run_worker()


if __name__ == "__main__":
    main()
