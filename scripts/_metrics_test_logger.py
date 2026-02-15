#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_metrics_test(
    *,
    test_name: str,
    payload: Dict[str, Any],
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append one test record as JSONL for audit/replay.
    This function must not raise to avoid breaking gate scripts.
    """
    try:
        out_dir = Path(os.getenv("METRICS_TEST_LOG_DIR", "artifacts/metrics_tests"))
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "test_name": str(test_name),
            "recorded_at": _utc_now_iso(),
            "window_start": window_start,
            "window_end": window_end,
            "payload": payload,
            "extra": extra or {},
        }
        log_file = out_dir / "metrics_tests.jsonl"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Logging should be best-effort and must never break metric scripts.
        return

