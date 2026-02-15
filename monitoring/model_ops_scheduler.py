#!/usr/bin/env python3
"""
Model Ops Scheduler
- Periodically triggers drift evaluation / auto-gate / rollback checks.
"""
from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TRACKS = [t.strip() for t in os.getenv("MODEL_OPS_TRACKS", "liquid").split(",") if t.strip()]
DRIFT_INTERVAL_SEC = int(os.getenv("DRIFT_INTERVAL_SEC", "900"))
GATE_INTERVAL_SEC = int(os.getenv("GATE_INTERVAL_SEC", "3600"))
ROLLBACK_INTERVAL_SEC = int(os.getenv("ROLLBACK_INTERVAL_SEC", "900"))
TIMEOUT_SEC = float(os.getenv("MODEL_OPS_TIMEOUT_SEC", "8"))


def _post(path: str, payload: dict) -> dict:
    url = f"{API_BASE}{path}"
    resp = requests.post(url, json=payload, timeout=TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def tick(track: str):
    now = datetime.now(timezone.utc).timestamp()
    if int(now) % DRIFT_INTERVAL_SEC == 0:
        try:
            out = _post("/api/v2/models/drift/evaluate", {"track": track})
            logger.info("drift track=%s action=%s metrics=%s", track, out.get("action"), out.get("metrics"))
        except Exception as exc:
            logger.error("drift tick failed track=%s err=%s", track, exc)

    if int(now) % GATE_INTERVAL_SEC == 0:
        try:
            out = _post("/api/v2/models/gate/auto-evaluate", {"track": track, "auto_promote": True})
            logger.info("gate track=%s passed=%s promoted=%s", track, out.get("passed"), out.get("promoted"))
        except Exception as exc:
            logger.error("gate tick failed track=%s err=%s", track, exc)

    if int(now) % ROLLBACK_INTERVAL_SEC == 0:
        try:
            out = _post(
                "/api/v2/models/rollback/check",
                {
                    "track": track,
                    "model_name": "liquid_ttm_ensemble" if track == "liquid" else "vc_survival_model",
                    "model_version": "v2.1",
                },
            )
            logger.info("rollback track=%s triggered=%s reason=%s", track, out.get("rollback_triggered"), out.get("reason"))
        except Exception as exc:
            logger.error("rollback tick failed track=%s err=%s", track, exc)


def main():
    logger.info("model-ops scheduler started tracks=%s", TRACKS)
    while True:
        for track in TRACKS:
            tick(track)
        time.sleep(1)


if __name__ == "__main__":
    main()
