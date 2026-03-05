from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from v2_repository import V2Repository


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str):
        return [p for p in shlex.split(x) if p]
    return []


@dataclass
class ManagedProcess:
    process_id: str
    task_type: str
    command: List[str]
    env_overrides: Dict[str, str]
    config_snapshot: Dict[str, Any]
    log_path: str
    metrics_path: Optional[str]
    account_id: Optional[str]
    track: str
    symbols: List[str]
    created_by: str
    auto_restart: bool = False
    max_restarts: int = 0
    restart_count: int = 0
    status: str = "created"
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    popen: Optional[subprocess.Popen] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_id": self.process_id,
            "task_type": self.task_type,
            "status": self.status,
            "pid": self.pid,
            "start_time": _iso(self.start_time),
            "end_time": _iso(self.end_time),
            "command": list(self.command),
            "env_overrides": dict(self.env_overrides),
            "config_snapshot": dict(self.config_snapshot),
            "log_path": self.log_path,
            "metrics_path": self.metrics_path,
            "account_id": self.account_id,
            "track": self.track,
            "symbols": list(self.symbols),
            "restart_count": self.restart_count,
            "auto_restart": self.auto_restart,
            "max_restarts": self.max_restarts,
            "exit_code": self.exit_code,
            "error": self.error,
            "created_by": self.created_by,
        }


class ProcessManager:
    def __init__(self, repo: V2Repository):
        self.repo = repo
        self._items: Dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()
        self._stop_flag = False
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="process-manager-monitor", daemon=True)
        self._monitor_thread.start()
        self.repo_root = Path(__file__).resolve().parents[1]
        self.python_bin = os.getenv("PYTHON_BIN", "python3")
        self.logs_dir = self.repo_root / "artifacts" / "ops" / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def shutdown(self) -> None:
        self._stop_flag = True
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)

    def _persist(self, item: ManagedProcess) -> None:
        try:
            self.repo.upsert_ops_process(item.process_id, item.to_dict())
        except Exception:
            pass

    def _event(self, process_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        try:
            self.repo.append_ops_process_event(process_id=process_id, event_type=event_type, payload=payload)
        except Exception:
            pass

    def _task_default_command(self, task_type: str, params: Dict[str, Any]) -> List[str]:
        t = str(task_type or "").upper()
        if t == "TRAIN_LIQUID":
            cmd = [self.python_bin, "training/train_liquid.py", "--model-id", str(params.get("model_id") or "liquid_main"), "--out-dir", str(params.get("out_dir") or "artifacts/models/liquid_main")]
            if params.get("symbols"):
                cmd.extend(["--symbols", str(params.get("symbols"))])
            return cmd
        if t == "TRAIN_VC":
            return [self.python_bin, "training/train_vc.py"]
        if t == "PAPER_TRADER":
            cmd = [self.python_bin, "monitoring/paper_trading_daemon.py", "--loop"]
            if params.get("symbols"):
                cmd.extend(["--symbols", str(params.get("symbols"))])
            return cmd
        if t == "LIVE_TRADER":
            # strict-only: current repo reuses paper trading daemon control chain for live toggle.
            cmd = [self.python_bin, "monitoring/paper_trading_daemon.py", "--loop"]
            if params.get("symbols"):
                cmd.extend(["--symbols", str(params.get("symbols"))])
            return cmd
        if t == "RECON_DAEMON":
            return [self.python_bin, "monitoring/reconciliation_daemon.py", "--loop"]
        if t == "CONTINUOUS_FINETUNE":
            return [self.python_bin, "monitoring/model_ops_scheduler.py"]
        if t == "ABLATION_EXPERIMENT":
            return [self.python_bin, "scripts/run_liquid_param_sweep.py"]
        raise ValueError(f"unsupported_task_type:{task_type}")

    def start(self, *, task_type: str, params: Dict[str, Any], created_by: str = "api") -> Dict[str, Any]:
        command = _as_list(params.get("command"))
        if not command:
            command = self._task_default_command(task_type, params)
        process_id = str(params.get("process_id") or f"proc-{uuid.uuid4().hex[:16]}")
        log_path = str(params.get("log_path") or (self.logs_dir / f"{process_id}.log"))
        env_overrides = {str(k): str(v) for k, v in dict(params.get("env_overrides") or {}).items()}
        auto_restart = bool(params.get("auto_restart") or False)
        max_restarts = int(params.get("max_restarts") or 0)
        symbols_raw = params.get("symbols")
        symbols: List[str]
        if isinstance(symbols_raw, list):
            symbols = [str(x).upper() for x in symbols_raw if str(x).strip()]
        else:
            symbols = [s.strip().upper() for s in str(symbols_raw or "").split(",") if s.strip()]
        item = ManagedProcess(
            process_id=process_id,
            task_type=str(task_type).upper(),
            command=command,
            env_overrides=env_overrides,
            config_snapshot=dict(params.get("config_snapshot") or {}),
            log_path=log_path,
            metrics_path=params.get("metrics_path"),
            account_id=(str(params.get("account_id")) if params.get("account_id") is not None else None),
            track=str(params.get("track") or "liquid"),
            symbols=symbols,
            created_by=str(created_by or "api"),
            auto_restart=auto_restart,
            max_restarts=max_restarts,
        )
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab", buffering=0) as lf:
            env = os.environ.copy()
            env.update(env_overrides)
            try:
                popen = subprocess.Popen(
                    command,
                    cwd=str(self.repo_root),
                    stdout=lf,
                    stderr=lf,
                    env=env,
                )
            except Exception as exc:
                item.status = "failed"
                item.error = str(exc)
                item.end_time = _utcnow()
                self._persist(item)
                self._event(item.process_id, "start_failed", {"error": str(exc), "command": command})
                raise
        item.popen = popen
        item.pid = int(popen.pid)
        item.status = "running"
        item.start_time = _utcnow()
        with self._lock:
            self._items[item.process_id] = item
        self._persist(item)
        self._event(item.process_id, "started", {"pid": item.pid, "command": command})
        return item.to_dict()

    def _terminate(self, item: ManagedProcess, *, force: bool = False) -> None:
        if item.popen is None:
            return
        if item.popen.poll() is not None:
            return
        try:
            if force:
                item.popen.kill()
            else:
                item.popen.terminate()
        except Exception:
            pass

    def stop(self, process_id: str, *, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            item = self._items.get(str(process_id))
        if item is None:
            raise ValueError("process_not_found")
        self._terminate(item, force=force)
        deadline = time.time() + 5.0
        while item.popen is not None and item.popen.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if item.popen is not None and item.popen.poll() is None and not force:
            self._terminate(item, force=True)
        if item.popen is not None:
            rc = item.popen.poll()
            item.exit_code = int(rc) if rc is not None else None
        item.status = "stopped"
        item.end_time = _utcnow()
        self._persist(item)
        self._event(item.process_id, "stopped", {"force": bool(force), "exit_code": item.exit_code})
        return item.to_dict()

    def restart(self, process_id: str, *, created_by: str = "api") -> Dict[str, Any]:
        with self._lock:
            item = self._items.get(str(process_id))
        if item is None:
            raise ValueError("process_not_found")
        self.stop(process_id, force=False)
        params = {
            "command": list(item.command),
            "env_overrides": dict(item.env_overrides),
            "config_snapshot": dict(item.config_snapshot),
            "log_path": item.log_path,
            "metrics_path": item.metrics_path,
            "account_id": item.account_id,
            "track": item.track,
            "symbols": list(item.symbols),
            "auto_restart": item.auto_restart,
            "max_restarts": item.max_restarts,
            "process_id": item.process_id,
        }
        out = self.start(task_type=item.task_type, params=params, created_by=created_by)
        self._event(item.process_id, "restarted", {"by": created_by})
        return out

    def list_processes(self, *, status: Optional[str] = None, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            rows = [x.to_dict() for x in self._items.values()]
        out: List[Dict[str, Any]] = []
        for r in rows:
            if status and str(r.get("status")) != str(status):
                continue
            if task_type and str(r.get("task_type")) != str(task_type).upper():
                continue
            out.append(r)
        out.sort(key=lambda x: str(x.get("process_id")))
        return out

    def get_process(self, process_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._items.get(str(process_id))
        return item.to_dict() if item else None

    def tail_logs(self, process_id: str, *, lines: int = 200) -> Dict[str, Any]:
        with self._lock:
            item = self._items.get(str(process_id))
        if item is None:
            raise ValueError("process_not_found")
        p = Path(item.log_path)
        if not p.exists():
            return {"process_id": process_id, "lines": []}
        raw = p.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = raw[-max(1, min(5000, int(lines))):]
        return {"process_id": process_id, "log_path": str(p), "lines": tail}

    def process_metrics(self, process_id: str) -> Dict[str, Any]:
        with self._lock:
            item = self._items.get(str(process_id))
        if item is None:
            raise ValueError("process_not_found")
        metrics = {
            "process_id": process_id,
            "status": item.status,
            "pid": item.pid,
            "cpu_percent": 0.0,
            "memory_rss": 0,
            "memory_vms": 0,
            "num_threads": 0,
            "open_files": 0,
        }
        if item.pid:
            try:
                proc = psutil.Process(item.pid)
                metrics.update(
                    {
                        "cpu_percent": float(proc.cpu_percent(interval=0.05)),
                        "memory_rss": int(proc.memory_info().rss),
                        "memory_vms": int(proc.memory_info().vms),
                        "num_threads": int(proc.num_threads()),
                        "open_files": len(proc.open_files()),
                    }
                )
            except Exception:
                pass
        return metrics

    def _monitor_loop(self) -> None:
        while not self._stop_flag:
            time.sleep(1.0)
            with self._lock:
                items = list(self._items.values())
            for item in items:
                p = item.popen
                if p is None:
                    continue
                rc = p.poll()
                if rc is None:
                    continue
                if item.status in {"finished", "failed", "stopped"}:
                    continue
                item.exit_code = int(rc)
                item.end_time = _utcnow()
                if rc == 0:
                    item.status = "finished"
                else:
                    item.status = "failed"
                    item.error = f"exit_code:{rc}"
                self._persist(item)
                self._event(item.process_id, "exited", {"exit_code": rc, "status": item.status})
                if item.status == "failed" and item.auto_restart and item.restart_count < item.max_restarts:
                    item.restart_count += 1
                    self._persist(item)
                    self._event(item.process_id, "auto_restart", {"restart_count": item.restart_count})
                    try:
                        with open(item.log_path, "ab", buffering=0) as lf:
                            env = os.environ.copy()
                            env.update(item.env_overrides)
                            popen = subprocess.Popen(
                                item.command,
                                cwd=str(self.repo_root),
                                stdout=lf,
                                stderr=lf,
                                env=env,
                            )
                        item.popen = popen
                        item.pid = int(popen.pid)
                        item.status = "running"
                        item.start_time = _utcnow()
                        item.end_time = None
                        item.exit_code = None
                        item.error = None
                        self._persist(item)
                        self._event(item.process_id, "auto_restart_started", {"pid": item.pid})
                    except Exception as exc:
                        item.status = "failed"
                        item.error = f"auto_restart_failed:{exc}"
                        self._persist(item)
                        self._event(item.process_id, "auto_restart_failed", {"error": str(exc)})
