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
WEEKLY_GATE_MODE = os.getenv("WEEKLY_GATE_MODE", "1").lower() in {"1", "true", "yes", "y"}
GATE_WEEKDAY = int(os.getenv("GATE_WEEKDAY", "1"))  # Monday=1
GATE_HOUR_UTC = int(os.getenv("GATE_HOUR_UTC", "2"))
ROLLOUT_AUTO_ADVANCE = os.getenv("ROLLOUT_AUTO_ADVANCE", "1").lower() in {"1", "true", "yes", "y"}
DRIFT_LOOKBACK_HOURS = int(os.getenv("DRIFT_LOOKBACK_HOURS", "24"))
DRIFT_REFERENCE_HOURS = int(os.getenv("DRIFT_REFERENCE_HOURS", "72"))
DRIFT_PSI_THRESHOLD = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.25"))
DRIFT_KS_THRESHOLD = float(os.getenv("DRIFT_KS_THRESHOLD", "0.20"))
GATE_MIN_IC = float(os.getenv("GATE_MIN_IC", "0.0"))
GATE_MIN_PNL_AFTER_COST = float(os.getenv("GATE_MIN_PNL_AFTER_COST", "0.0"))
GATE_MAX_DRAWDOWN = float(os.getenv("GATE_MAX_DRAWDOWN", "0.2"))
GATE_WINDOWS = int(os.getenv("GATE_WINDOWS", "3"))
ROLLBACK_MIN_HIT_RATE = float(os.getenv("ROLLBACK_MIN_HIT_RATE", "0.4"))
ROLLBACK_MAX_DRAWDOWN = float(os.getenv("ROLLBACK_MAX_DRAWDOWN", "0.25"))
ROLLBACK_MAX_RECENT_LOSSES = int(os.getenv("ROLLBACK_MAX_RECENT_LOSSES", "5"))
ROLLOUT_MIN_HIT_RATE = float(os.getenv("ROLLOUT_MIN_HIT_RATE", "0.45"))
ROLLOUT_MIN_PNL_AFTER_COST = float(os.getenv("ROLLOUT_MIN_PNL_AFTER_COST", "0.0"))
ROLLOUT_MAX_DRAWDOWN = float(os.getenv("ROLLOUT_MAX_DRAWDOWN", "0.25"))
ROLLOUT_WINDOWS = int(os.getenv("ROLLOUT_WINDOWS", "3"))
PARITY_MAX_DEVIATION = float(os.getenv("PARITY_MAX_DEVIATION", "0.10"))
PARITY_MIN_COMPLETED_RUNS = int(os.getenv("PARITY_MIN_COMPLETED_RUNS", "5"))
MODEL_NAME_LIQUID = os.getenv("MODEL_NAME_LIQUID", "liquid_ttm_ensemble")
MODEL_VERSION_LIQUID = os.getenv("MODEL_VERSION_LIQUID", "v2.1")
MODEL_NAME_VC = os.getenv("MODEL_NAME_VC", "vc_survival_model")
MODEL_VERSION_VC = os.getenv("MODEL_VERSION_VC", "v2.1")


def _track_model(track: str) -> tuple[str, str]:
    if track == "liquid":
        return MODEL_NAME_LIQUID, MODEL_VERSION_LIQUID
    return MODEL_NAME_VC, MODEL_VERSION_VC


def _audit_log(track: str, action: str, window: dict, thresholds: dict, decision: dict) -> None:
    payload = {
        "track": track,
        "action": action,
        "window": window,
        "thresholds": thresholds,
        "decision": decision,
    }
    logger.info("audit who=system source=scheduler track=%s action=%s payload=%s", track, action, payload)
    try:
        _post("/api/v2/models/audit/log", payload)
    except Exception:
        logger.warning("audit persist failed track=%s action=%s", track, action)


def _post(path: str, payload: dict) -> dict:
    url = f"{API_BASE}{path}"
    resp = requests.post(url, json=payload, timeout=TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def _get(path: str, params: dict | None = None) -> dict:
    url = f"{API_BASE}{path}"
    resp = requests.get(url, params=params or {}, timeout=TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def _run_drift(track: str):
    try:
        payload = {
            "track": track,
            "lookback_hours": DRIFT_LOOKBACK_HOURS,
            "reference_hours": DRIFT_REFERENCE_HOURS,
            "psi_threshold": DRIFT_PSI_THRESHOLD,
            "ks_threshold": DRIFT_KS_THRESHOLD,
        }
        out = _post("/api/v2/models/drift/evaluate", payload)
        logger.info("drift track=%s action=%s metrics=%s", track, out.get("action"), out.get("metrics"))
        _audit_log(
            track=track,
            action="drift_evaluate",
            window={"lookback_hours": DRIFT_LOOKBACK_HOURS, "reference_hours": DRIFT_REFERENCE_HOURS},
            thresholds={"psi_threshold": DRIFT_PSI_THRESHOLD, "ks_threshold": DRIFT_KS_THRESHOLD},
            decision={"action": out.get("action"), "reason": out.get("reason"), "drift_detected": out.get("drift_detected")},
        )
    except Exception as exc:
        logger.error("drift tick failed track=%s err=%s", track, exc)


def _run_gate(track: str):
    try:
        model_name, model_version = _track_model(track)
        payload = {
            "track": track,
            "model_name": model_name,
            "model_version": model_version,
            "min_ic": GATE_MIN_IC,
            "min_pnl_after_cost": GATE_MIN_PNL_AFTER_COST,
            "max_drawdown": GATE_MAX_DRAWDOWN,
            "windows": GATE_WINDOWS,
            "auto_promote": True,
        }
        out = _post("/api/v2/models/gate/auto-evaluate", payload)
        logger.info("gate track=%s passed=%s promoted=%s", track, out.get("passed"), out.get("promoted"))
        _audit_log(
            track=track,
            action="auto_gate",
            window={"windows": GATE_WINDOWS},
            thresholds={
                "min_ic": GATE_MIN_IC,
                "min_pnl_after_cost": GATE_MIN_PNL_AFTER_COST,
                "max_drawdown": GATE_MAX_DRAWDOWN,
            },
            decision={"passed": out.get("passed"), "promoted": out.get("promoted"), "reason": out.get("reason")},
        )
    except Exception as exc:
        logger.error("gate tick failed track=%s err=%s", track, exc)

    try:
        out = _post(
            "/api/v2/models/parity/check",
            {
                "track": track,
                "max_deviation": PARITY_MAX_DEVIATION,
                "min_completed_runs": PARITY_MIN_COMPLETED_RUNS,
            },
        )
        logger.info(
            "parity track=%s status=%s passed=%s rel30=%s",
            track,
            out.get("status"),
            out.get("passed"),
            ((out.get("windows") or {}).get("30d") or {}).get("relative_deviation"),
        )
        _audit_log(
            track=track,
            action="parity_check",
            window={"alert_window": "7d", "gate_window": "30d"},
            thresholds={"max_deviation": PARITY_MAX_DEVIATION, "min_completed_runs": PARITY_MIN_COMPLETED_RUNS},
            decision={"status": out.get("status"), "passed": out.get("passed")},
        )
    except Exception as exc:
        logger.error("parity tick failed track=%s err=%s", track, exc)

    if not ROLLOUT_AUTO_ADVANCE:
        return
    try:
        model_name, model_version = _track_model(track)
        state = _get("/api/v2/models/rollout/state", {"track": track})
        current_stage = int(state.get("stage_pct") or 10)
        ladder = {10: 30, 30: 100, 100: 100}
        if current_stage not in ladder:
            current_stage = 10
        next_stage = ladder[current_stage]
        if current_stage >= 100 or next_stage == current_stage:
            logger.info("rollout track=%s skipped reason=already_max_stage stage=%s", track, current_stage)
            _audit_log(
                track=track,
                action="rollout_advance",
                window={"windows": ROLLOUT_WINDOWS},
                thresholds={
                    "min_hit_rate": ROLLOUT_MIN_HIT_RATE,
                    "min_pnl_after_cost": ROLLOUT_MIN_PNL_AFTER_COST,
                    "max_drawdown": ROLLOUT_MAX_DRAWDOWN,
                },
                decision={"promoted": False, "next_stage_pct": current_stage, "reason": "already_max_stage"},
            )
            return
        out = _post(
            "/api/v2/models/rollout/advance",
            {
                "track": track,
                "model_name": model_name,
                "model_version": model_version,
                "current_stage_pct": current_stage,
                "next_stage_pct": next_stage,
                "min_hit_rate": ROLLOUT_MIN_HIT_RATE,
                "min_pnl_after_cost": ROLLOUT_MIN_PNL_AFTER_COST,
                "max_drawdown": ROLLOUT_MAX_DRAWDOWN,
                "windows": ROLLOUT_WINDOWS,
            },
        )
        logger.info(
            "rollout track=%s promoted=%s next_stage=%s reason=%s",
            track,
            out.get("promoted"),
            out.get("next_stage_pct"),
            out.get("reason"),
        )
        _audit_log(
            track=track,
            action="rollout_advance",
            window={"windows": ROLLOUT_WINDOWS},
            thresholds={
                "min_hit_rate": ROLLOUT_MIN_HIT_RATE,
                "min_pnl_after_cost": ROLLOUT_MIN_PNL_AFTER_COST,
                "max_drawdown": ROLLOUT_MAX_DRAWDOWN,
            },
            decision={"promoted": out.get("promoted"), "next_stage_pct": out.get("next_stage_pct"), "reason": out.get("reason")},
        )
    except Exception as exc:
        logger.error("rollout tick failed track=%s err=%s", track, exc)


def _run_rollback(track: str):
    try:
        model_name, model_version = _track_model(track)
        out = _post(
            "/api/v2/models/rollback/check",
            {
                "track": track,
                "model_name": model_name,
                "model_version": model_version,
                "max_recent_losses": ROLLBACK_MAX_RECENT_LOSSES,
                "min_recent_hit_rate": ROLLBACK_MIN_HIT_RATE,
                "max_recent_drawdown": ROLLBACK_MAX_DRAWDOWN,
            },
        )
        logger.info("rollback track=%s triggered=%s reason=%s", track, out.get("rollback_triggered"), out.get("reason"))
        _audit_log(
            track=track,
            action="rollback_check",
            window={"max_recent_losses": ROLLBACK_MAX_RECENT_LOSSES},
            thresholds={"min_recent_hit_rate": ROLLBACK_MIN_HIT_RATE, "max_recent_drawdown": ROLLBACK_MAX_DRAWDOWN},
            decision={
                "rollback_triggered": out.get("rollback_triggered"),
                "windows_failed": out.get("windows_failed"),
                "trigger_rule": out.get("trigger_rule"),
                "reason": out.get("reason"),
            },
        )
    except Exception as exc:
        logger.error("rollback tick failed track=%s err=%s", track, exc)


def main():
    logger.info("model-ops scheduler started tracks=%s", TRACKS)
    now = datetime.now(timezone.utc).timestamp()
    next_drift = {track: now for track in TRACKS}
    next_gate = {track: now for track in TRACKS}
    gate_week_marker = {track: "" for track in TRACKS}
    next_rollback = {track: now for track in TRACKS}
    while True:
        now = datetime.now(timezone.utc).timestamp()
        for track in TRACKS:
            if now >= next_drift[track]:
                _run_drift(track)
                next_drift[track] = now + DRIFT_INTERVAL_SEC
            if now >= next_gate[track]:
                if WEEKLY_GATE_MODE:
                    dt = datetime.now(timezone.utc)
                    marker = f"{dt.isocalendar().year}-W{dt.isocalendar().week}"
                    is_gate_slot = dt.isoweekday() == GATE_WEEKDAY and dt.hour == GATE_HOUR_UTC and dt.minute < 5
                    if is_gate_slot and gate_week_marker[track] != marker:
                        _run_gate(track)
                        gate_week_marker[track] = marker
                    next_gate[track] = now + 60
                else:
                    _run_gate(track)
                    next_gate[track] = now + GATE_INTERVAL_SEC
            if now >= next_rollback[track]:
                _run_rollback(track)
                next_rollback[track] = now + ROLLBACK_INTERVAL_SEC
        time.sleep(1)


if __name__ == "__main__":
    main()
