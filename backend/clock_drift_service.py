from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

import requests

from v2_repository import V2Repository


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class ClockDriftService:
    def __init__(self, repo: V2Repository):
        self.repo = repo

    def _level(self, drift_ms: Optional[float], err: Optional[str]) -> str:
        if err is not None:
            return "red"
        if drift_ms is None:
            return "red"
        yellow = float(__import__("os").getenv("CLOCK_DRIFT_YELLOW_MS", "500"))
        red = float(__import__("os").getenv("CLOCK_DRIFT_RED_MS", "1500"))
        v = abs(float(drift_ms))
        if v > red:
            return "red"
        if v > yellow:
            return "yellow"
        return "green"

    def probe(self, *, source_url: str = "https://api.bitget.com/api/v2/public/time", timeout_sec: float = 5.0, persist: bool = True) -> Dict[str, Any]:
        local = _utcnow()
        remote: Optional[datetime] = None
        err: Optional[str] = None
        try:
            r = requests.head(source_url, timeout=timeout_sec)
            date_hdr = str(r.headers.get("Date") or "").strip()
            if date_hdr:
                remote = parsedate_to_datetime(date_hdr).astimezone(timezone.utc)
            else:
                rr = requests.get(source_url, timeout=timeout_sec)
                date2 = str(rr.headers.get("Date") or "").strip()
                if date2:
                    remote = parsedate_to_datetime(date2).astimezone(timezone.utc)
                else:
                    err = "http_date_header_missing"
        except Exception as exc:
            err = f"clock_probe_error:{exc}"

        drift_ms: Optional[float]
        if remote is not None:
            drift_ms = (local - remote).total_seconds() * 1000.0
        else:
            drift_ms = None
        level = self._level(drift_ms, err)
        out = {
            "source": source_url,
            "local_utc": _iso(local),
            "remote_utc": _iso(remote),
            "drift_ms": (round(float(drift_ms), 3) if drift_ms is not None else None),
            "level": level,
            "error": err,
        }
        if persist:
            self.repo.save_clock_drift_status(
                source=source_url,
                local_utc=local,
                remote_utc=remote,
                drift_ms=drift_ms,
                level=level,
                error=err,
                payload=out,
            )
        return out

    def latest(self) -> Dict[str, Any]:
        row = self.repo.get_latest_clock_drift_status()
        if not row:
            return {"status": "missing"}
        return {
            "status": "ok",
            "source": row.get("source"),
            "local_utc": _iso(row.get("local_utc")),
            "remote_utc": _iso(row.get("remote_utc")),
            "drift_ms": float(row.get("drift_ms") or 0.0) if row.get("drift_ms") is not None else None,
            "level": str(row.get("level") or "red"),
            "error": row.get("error"),
            "ts": _iso(row.get("ts")),
        }

    def assert_live_safe(self, *, max_level: str = "yellow") -> Dict[str, Any]:
        row = self.latest()
        if row.get("status") != "ok":
            raise RuntimeError("clock_drift_missing")
        level = str(row.get("level") or "red").lower()
        order = {"green": 1, "yellow": 2, "red": 3}
        cur = order.get(level, 99)
        lim = order.get(str(max_level).lower(), 2)
        if cur > lim:
            raise RuntimeError(f"clock_drift_level_exceeded:{level}")
        return row
