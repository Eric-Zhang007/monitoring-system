from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote


def mask_sensitive(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if any(x in kl for x in ("secret", "password", "passphrase", "api_key", "token")):
                out[str(k)] = "***"
            else:
                out[str(k)] = mask_sensitive(v)
        return out
    if isinstance(obj, list):
        return [mask_sensitive(x) for x in obj]
    return obj


def proxy_url_from_profile(profile: Dict[str, Any], password_plain: Optional[str] = None) -> str:
    ptype = str(profile.get("proxy_type") or "").strip().lower()
    if ptype not in {"http", "https", "socks5"}:
        raise ValueError(f"unsupported_proxy_type:{ptype}")
    host = str(profile.get("host") or "").strip()
    port = int(profile.get("port") or 0)
    if not host or port <= 0:
        raise ValueError("proxy_host_port_required")
    user = str(profile.get("username") or "").strip()
    auth = ""
    if user:
        pwd = str(password_plain or "")
        auth = f"{quote(user, safe='')}:{quote(pwd, safe='')}@"
    return f"{ptype}://{auth}{host}:{port}"


def requests_proxies(proxy_url: str) -> Dict[str, str]:
    return {
        "http": proxy_url,
        "https": proxy_url,
    }


def proxy_env_overrides(proxy_url: str) -> Dict[str, str]:
    return {
        "HTTP_PROXY": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "ALL_PROXY": proxy_url,
    }


def ws_proxy_supported(profile: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    ptype = str(profile.get("proxy_type") or "").strip().lower()
    if ptype in {"http", "https"}:
        return True, None
    if ptype == "socks5":
        return False, "ws_socks5_not_supported_by_default_client_use_http_connect_proxy"
    return False, f"unsupported_proxy_type:{ptype}"
