#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import requests


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate Bitget live connectivity")
    ap.add_argument("--require-creds", action="store_true")
    args = ap.parse_args()

    key = os.getenv("BITGET_API_KEY", "").strip()
    secret = os.getenv("BITGET_API_SECRET", "").strip()
    passphrase = os.getenv("BITGET_API_PASSPHRASE", "").strip()
    base_url = os.getenv("BITGET_BASE_URL", "https://api.bitget.com").strip().rstrip("/")

    public_ok = False
    status_code = 0
    try:
        resp = requests.get(f"{base_url}/api/v2/public/time", timeout=8)
        status_code = resp.status_code
        public_ok = status_code < 400
    except Exception:
        public_ok = False

    if not (key and secret and passphrase):
        out = {
            "status": "skipped",
            "public_api_ok": public_ok,
            "public_http_status": status_code,
            "reason": "missing_credentials",
            "required_env": ["BITGET_API_KEY", "BITGET_API_SECRET", "BITGET_API_PASSPHRASE"],
        }
        print(json.dumps(out, ensure_ascii=False))
        return 2 if args.require_creds else 0

    # Minimal auth sanity is delegated to runtime adapter; here only report credential presence + public API.
    out = {
        "status": "ok" if public_ok else "failed",
        "public_api_ok": public_ok,
        "public_http_status": status_code,
        "credentials_present": True,
        "note": "credentials detected; authenticated order checks should run in controlled environment",
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0 if public_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
