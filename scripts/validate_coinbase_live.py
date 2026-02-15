#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import requests


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate Coinbase live connectivity")
    ap.add_argument("--require-creds", action="store_true")
    args = ap.parse_args()

    has_key = bool(os.getenv("COINBASE_API_KEY", "").strip())
    has_secret = bool(os.getenv("COINBASE_API_SECRET", "").strip())

    # Public endpoint connectivity (no auth).
    public_ok = False
    status_code = 0
    try:
        resp = requests.get("https://api.coinbase.com/v2/time", timeout=8)
        status_code = resp.status_code
        public_ok = status_code < 400
    except Exception:
        public_ok = False

    if not (has_key and has_secret):
        out = {
            "status": "skipped",
            "public_api_ok": public_ok,
            "public_http_status": status_code,
            "reason": "missing_credentials",
            "required_env": ["COINBASE_API_KEY", "COINBASE_API_SECRET"],
        }
        print(json.dumps(out, ensure_ascii=False))
        return 2 if args.require_creds else 0

    out = {
        "status": "ok" if public_ok else "failed",
        "public_api_ok": public_ok,
        "public_http_status": status_code,
        "credentials_present": True,
        "note": "credentials detected; authenticated trade checks should be run in controlled environment",
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0 if public_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
