#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import logging
import os
import sys
import traceback
from typing import Any, Dict

sys.path.append("/app")

from task_queue import claim_next_task, set_task_status
from schemas_v2 import BacktestRunRequest
from v2_repository import V2Repository
from v2_router import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")
POLL_TIMEOUT_SEC = int(os.getenv("TASK_WORKER_POLL_TIMEOUT_SEC", "5"))


def _run_backtest_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    req = BacktestRunRequest(**payload)
    resp = asyncio.run(run_backtest(req))
    return resp.model_dump()


def _run_pnl_job(payload: Dict[str, Any]) -> Dict[str, Any]:
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


def main() -> None:
    logger.info("task worker started")
    while True:
        task = claim_next_task(timeout_seconds=POLL_TIMEOUT_SEC)
        if not task:
            continue
        task_id = str(task.get("task_id") or "")
        task_type = str(task.get("task_type") or "")
        payload = task.get("payload") or {}
        if not task_id or not task_type:
            continue
        set_task_status(task_id, "running")
        try:
            if task_type == "backtest_run":
                result = _run_backtest_job(payload)
            elif task_type == "pnl_attribution":
                result = _run_pnl_job(payload)
            else:
                raise ValueError(f"unsupported_task_type:{task_type}")
            set_task_status(task_id, "completed", result=result)
            logger.info("task completed id=%s type=%s", task_id, task_type)
        except Exception as exc:
            set_task_status(task_id, "failed", error=f"{exc}")
            logger.error("task failed id=%s type=%s err=%s\n%s", task_id, task_type, exc, traceback.format_exc())


if __name__ == "__main__":
    main()
