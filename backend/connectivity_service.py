from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from typing import Any, Dict, Optional

import requests
try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    websockets = None  # type: ignore[assignment]

from proxy_utils import proxy_env_overrides, proxy_url_from_profile, requests_proxies, ws_proxy_supported
from v2_repository import V2Repository


VENUE_ENDPOINTS = {
    "bitget": {
        "rest_url": "https://api.bitget.com/api/v2/public/time",
        "ws_url": "wss://ws.bitget.com/v2/ws/public",
    },
    "coinbase": {
        "rest_url": "https://api.coinbase.com/v2/time",
        "ws_url": "wss://ws-feed.exchange.coinbase.com",
    },
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConnectivityService:
    def __init__(self, repo: V2Repository, secrets_manager: Optional[Any] = None):
        self.repo = repo
        self.secrets_manager = secrets_manager

    def _resolve_proxy_context(
        self,
        *,
        proxy_profile_id: Optional[str] = None,
        process_id: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        profile: Optional[Dict[str, Any]] = None
        if proxy_profile_id:
            profile = self.repo.get_proxy_profile_secret_row(str(proxy_profile_id))
        else:
            profile = self.repo.resolve_proxy_profile_for_target(process_id=process_id, account_id=account_id)
        if not profile:
            return {"profile": None, "proxy_url": None, "proxy_env": {}, "ws_supported": True, "ws_error": None}
        password_plain = ""
        if str(profile.get("password_enc") or "").strip():
            if not self.secrets_manager:
                raise RuntimeError("secrets_manager_required_for_proxy_password_decrypt")
            password_plain = self.secrets_manager.decrypt(str(profile.get("password_enc")))
        proxy_url = proxy_url_from_profile(profile, password_plain=password_plain)
        ws_ok, ws_error = ws_proxy_supported(profile)
        return {
            "profile": profile,
            "proxy_url": proxy_url,
            "proxy_env": proxy_env_overrides(proxy_url),
            "ws_supported": ws_ok,
            "ws_error": ws_error,
        }

    async def _probe_ws(self, ws_url: str, timeout_sec: float) -> bool:
        if websockets is not None:
            try:
                async with websockets.connect(ws_url, open_timeout=timeout_sec, close_timeout=timeout_sec):
                    return True
            except Exception:
                return False
        parsed = urlparse(str(ws_url))
        host = str(parsed.hostname or "").strip()
        if not host:
            return False
        port = int(parsed.port or (443 if parsed.scheme == "wss" else 80))
        ssl_ctx = True if parsed.scheme == "wss" else None
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host=host, port=port, ssl=ssl_ctx), timeout=timeout_sec)
            _ = reader
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    def probe(
        self,
        *,
        venue: str,
        proxy_profile_id: Optional[str] = None,
        process_id: Optional[str] = None,
        account_id: Optional[str] = None,
        timeout_sec: float = 5.0,
        persist: bool = True,
    ) -> Dict[str, Any]:
        venue_key = str(venue or "").strip().lower()
        cfg = VENUE_ENDPOINTS.get(venue_key)
        if not cfg:
            raise ValueError(f"unsupported_venue:{venue_key}")
        ctx = self._resolve_proxy_context(proxy_profile_id=proxy_profile_id, process_id=process_id, account_id=account_id)
        proxy_url = ctx.get("proxy_url")
        proxies = requests_proxies(proxy_url) if proxy_url else None
        profile = ctx.get("profile")

        start = time.perf_counter()
        rest_ok = False
        ws_ok = False
        err = None

        try:
            r = requests.get(cfg["rest_url"], timeout=timeout_sec, proxies=proxies)
            if int(r.status_code) in {200, 201, 202, 204}:
                rest_ok = True
            else:
                err = f"rest_status:{r.status_code}"
        except Exception as exc:
            err = f"rest_error:{exc}"

        if ctx.get("ws_supported", True):
            ws_ok = asyncio.run(self._probe_ws(cfg["ws_url"], timeout_sec=timeout_sec))
            if not ws_ok and err is None:
                err = "ws_probe_failed"
        else:
            ws_ok = False
            err = str(ctx.get("ws_error") or "ws_proxy_unsupported")

        latency_ms = (time.perf_counter() - start) * 1000.0
        payload = {
            "venue": venue_key,
            "rest_url": cfg["rest_url"],
            "ws_url": cfg["ws_url"],
            "proxy_profile_id": profile.get("profile_id") if isinstance(profile, dict) else None,
            "proxy_type": profile.get("proxy_type") if isinstance(profile, dict) else None,
            "account_id": account_id,
            "process_id": process_id,
        }
        out = {
            "venue": venue_key,
            "rest_ok": bool(rest_ok),
            "ws_ok": bool(ws_ok),
            "latency_ms": round(float(latency_ms), 3),
            "error": err,
            "using_proxy_profile": (profile.get("profile_id") if isinstance(profile, dict) else None),
            "payload": payload,
            "ts": _utcnow().isoformat().replace("+00:00", "Z"),
        }
        if persist:
            self.repo.save_venue_connectivity_status(
                venue=venue_key,
                rest_ok=bool(rest_ok),
                ws_ok=bool(ws_ok),
                latency_ms=float(latency_ms),
                error=err,
                using_proxy_profile=out["using_proxy_profile"],
                payload=payload,
            )
        return out

    def latest(self, *, venue: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
        rows = self.repo.list_venue_connectivity_status(venue=venue, limit=limit)
        return {"items": rows, "count": len(rows)}

    def assert_live_reachable(self, *, venue: str, max_age_sec: int = 120, require_ws: bool = True) -> Dict[str, Any]:
        row = self.repo.get_latest_venue_connectivity(venue)
        if not row:
            raise RuntimeError(f"connectivity_missing:{venue}")
        ts = row.get("ts")
        if isinstance(ts, datetime):
            age_sec = max(0.0, (_utcnow() - ts.astimezone(timezone.utc)).total_seconds())
        else:
            age_sec = 1e9
        if age_sec > float(max_age_sec):
            raise RuntimeError(f"connectivity_stale:{venue}:{age_sec:.1f}s")
        if not bool(row.get("rest_ok")):
            raise RuntimeError(f"connectivity_rest_unreachable:{venue}")
        if require_ws and not bool(row.get("ws_ok")):
            raise RuntimeError(f"connectivity_ws_unreachable:{venue}")
        return dict(row)
