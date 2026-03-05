#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
RUN_BG = ROOT / "scripts" / "run_bg_task.sh"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)


def _task_running(task_name: str) -> tuple[bool, str]:
    p = _run_cmd([str(RUN_BG), "status", task_name])
    text = (p.stdout or "") + (p.stderr or "")
    return (p.returncode == 0 and "running" in text.lower()), text.strip()


def _parse_checkpoint_updated_at(checkpoint_file: Path) -> Optional[datetime]:
    if not checkpoint_file.exists():
        return None
    try:
        obj = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        raw = str(obj.get("updated_at") or "").strip()
        if raw:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass
    st = checkpoint_file.stat()
    return datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)


def _checkpoint_summary(checkpoint_file: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"checkpoint_file": str(checkpoint_file), "exists": checkpoint_file.exists()}
    if not checkpoint_file.exists():
        return out
    try:
        obj = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        cc = obj.get("completed_chunks") or {}
        out["updated_at"] = obj.get("updated_at")
        out["completed_chunks"] = int(len(cc))
        if cc:
            k = sorted(cc.keys())[-1]
            out["last_chunk"] = k
            out["last_chunk_rows"] = int((cc.get(k) or {}).get("rows") or 0)
            out["last_chunk_completed_at"] = (cc.get(k) or {}).get("completed_at")
    except Exception as exc:
        out["parse_error"] = str(exc)
    return out


def _restart_task(task_name: str, start_cmd: str) -> Dict[str, Any]:
    stop_p = _run_cmd([str(RUN_BG), "stop", task_name])
    start_p = _run_cmd([str(RUN_BG), "start", task_name, "--", "bash", "-lc", start_cmd])
    return {
        "stop_returncode": int(stop_p.returncode),
        "stop_output": ((stop_p.stdout or "") + (stop_p.stderr or "")).strip(),
        "start_returncode": int(start_p.returncode),
        "start_output": ((start_p.stdout or "") + (start_p.stderr or "")).strip(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Watchdog for long-running ingestion task with stale checkpoint restart")
    ap.add_argument("--task-name", default=os.getenv("WATCHDOG_TASK_NAME", "full_history_mtf"))
    ap.add_argument("--checkpoint-file", required=True)
    ap.add_argument("--start-cmd", required=True, help="Command body used with `bash -lc` when restart is needed")
    ap.add_argument("--poll-seconds", type=int, default=int(os.getenv("WATCHDOG_POLL_SECONDS", "60")))
    ap.add_argument("--stale-seconds", type=int, default=int(os.getenv("WATCHDOG_STALE_SECONDS", "1800")))
    ap.add_argument("--max-restarts", type=int, default=int(os.getenv("WATCHDOG_MAX_RESTARTS", "10")))
    ap.add_argument("--max-runtime-seconds", type=int, default=int(os.getenv("WATCHDOG_MAX_RUNTIME_SECONDS", "0")))
    ap.add_argument("--status-file", default=os.getenv("WATCHDOG_STATUS_FILE", "artifacts/runtime/bg/status/ingest_watchdog_status.json"))
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    checkpoint_file = Path(str(args.checkpoint_file))
    status_file = Path(str(args.status_file))
    started_monotonic = time.monotonic()
    restarts = 0
    missing_consecutive = 0

    while True:
        now = datetime.now(timezone.utc)
        running, status_text = _task_running(str(args.task_name))
        updated_at = _parse_checkpoint_updated_at(checkpoint_file)
        stale = False
        stale_age = None
        if updated_at is not None:
            stale_age = (now - updated_at).total_seconds()
            stale = stale_age > float(max(30, int(args.stale_seconds)))

        action = "none"
        restart_info: Dict[str, Any] = {}
        if not running:
            missing_consecutive += 1
        else:
            missing_consecutive = 0

        if (not running and missing_consecutive >= 2) or (running and stale):
            if restarts >= int(args.max_restarts):
                payload = {
                    "status": "failed",
                    "updated_at": _now_iso(),
                    "task_name": str(args.task_name),
                    "reason": "max_restarts_exceeded",
                    "restarts": int(restarts),
                    "running": bool(running),
                    "status_text": status_text,
                    "checkpoint": _checkpoint_summary(checkpoint_file),
                }
                _save_json(status_file, payload)
                raise RuntimeError("watchdog_max_restarts_exceeded")
            restarts += 1
            action = "restart"
            restart_info = _restart_task(str(args.task_name), str(args.start_cmd))
            missing_consecutive = 0

        payload = {
            "status": "ok",
            "updated_at": _now_iso(),
            "task_name": str(args.task_name),
            "running": bool(running),
            "status_text": status_text,
            "stale": bool(stale),
            "stale_age_seconds": None if stale_age is None else round(float(stale_age), 3),
            "restarts": int(restarts),
            "action": action,
            "restart_info": restart_info,
            "checkpoint": _checkpoint_summary(checkpoint_file),
        }
        _save_json(status_file, payload)

        if bool(args.once):
            return 0
        if int(args.max_runtime_seconds) > 0 and (time.monotonic() - started_monotonic) >= int(args.max_runtime_seconds):
            return 0
        time.sleep(max(5, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
