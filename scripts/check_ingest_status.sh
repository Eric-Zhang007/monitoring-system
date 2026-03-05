#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASK_NAME="${1:-${TASK_NAME:-full_history_mtf}}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-$(ls -t artifacts/checkpoints/all_timeframes/ingest_perp_1m_*.json 2>/dev/null | head -n 1)}"
PROGRESS_FILE="${PROGRESS_FILE:-artifacts/runtime/bg/status/full_history_progress.json}"
LAST_EXIT_FILE="${LAST_EXIT_FILE:-artifacts/runtime/bg/status/full_history_last_exit.json}"
WATCHDOG_FILE="${WATCHDOG_FILE:-artifacts/runtime/bg/status/ingest_watchdog_status.json}"

printf 'task=%s\n' "${TASK_NAME}"
if [[ -x scripts/run_bg_task.sh ]]; then
  scripts/run_bg_task.sh status "${TASK_NAME}" || true
fi

python3 - <<'PY' "${CHECKPOINT_FILE}" "${PROGRESS_FILE}" "${LAST_EXIT_FILE}" "${WATCHDOG_FILE}"
from __future__ import annotations
import json
import sys
from pathlib import Path

ckpt, progress, last_exit, watchdog = [Path(x) for x in sys.argv[1:5]]

def show_json(path: Path, label: str) -> None:
    print(f"--- {label}: {path}")
    if not path.exists():
        print("missing")
        return
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"invalid_json: {exc}")
        return
    if label == "checkpoint":
        cc = obj.get("completed_chunks") or {}
        print("updated_at=", obj.get("updated_at"))
        print("completed_chunks=", len(cc))
        if cc:
            k = sorted(cc.keys())[-1]
            v = cc.get(k) or {}
            print("last_chunk=", k)
            print("last_rows=", v.get("rows"))
            print("last_completed_at=", v.get("completed_at"))
    else:
        keys = [
            "kind",
            "status",
            "updated_at",
            "task_name",
            "running",
            "stale",
            "stale_age_seconds",
            "restarts",
            "action",
            "exit_code",
            "current_timeframe",
        ]
        for k in keys:
            if k in obj:
                print(f"{k}=", obj.get(k))

show_json(ckpt, "checkpoint")
show_json(progress, "progress")
show_json(last_exit, "last_exit")
show_json(watchdog, "watchdog")
PY
