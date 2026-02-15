#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def _json_cmd(cmd: List[str]) -> Dict[str, Any]:
    code, out, _ = _run(cmd)
    if code != 0 or not out:
        return {}
    try:
        return json.loads(out)
    except Exception:
        return {}


def _psql(sql: str) -> tuple[int, str, str]:
    return _run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "monitor",
            "-d",
            "monitor",
            "-c",
            sql,
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy validated GPU model bundle to active_model_state")
    ap.add_argument("--track", default="liquid")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--model-version", required=True)
    ap.add_argument("--artifact-path", default="")
    ap.add_argument("--skip-readiness", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    artifact_path = Path(args.artifact_path) if args.artifact_path else None
    if artifact_path is not None and not artifact_path.exists():
        raise SystemExit(f"artifact not found: {artifact_path}")

    readiness = {}
    if not bool(args.skip_readiness):
        readiness = _json_cmd(["python3", "scripts/check_gpu_cutover_readiness.py"])
        if not bool(readiness.get("ready_for_gpu_cutover")):
            print(json.dumps({"status": "blocked", "reason": "readiness_failed", "readiness": readiness}, ensure_ascii=False))
            return 2

    meta = {
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(artifact_path) if artifact_path else "",
        "source": "deploy_gpu_bundle.py",
        "readiness": readiness,
    }
    sql = f"""
    INSERT INTO active_model_state (
      track, active_model_name, active_model_version,
      previous_model_name, previous_model_version, status, metadata, updated_at
    )
    VALUES (
      '{str(args.track).strip().lower()}',
      '{str(args.model_name).strip()}',
      '{str(args.model_version).strip()}',
      NULL, NULL, 'active', '{json.dumps(meta, ensure_ascii=False)}'::jsonb, NOW()
    )
    ON CONFLICT (track) DO UPDATE SET
      previous_model_name = active_model_state.active_model_name,
      previous_model_version = active_model_state.active_model_version,
      active_model_name = EXCLUDED.active_model_name,
      active_model_version = EXCLUDED.active_model_version,
      status = 'active',
      metadata = EXCLUDED.metadata,
      updated_at = NOW();
    """

    if bool(args.dry_run):
        print(json.dumps({"status": "dry_run", "sql": sql, "metadata": meta}, ensure_ascii=False))
        return 0
    code, out, err = _psql(sql)
    payload = {
        "status": "ok" if code == 0 else "failed",
        "track": str(args.track).strip().lower(),
        "model_name": str(args.model_name).strip(),
        "model_version": str(args.model_version).strip(),
        "artifact_path": str(artifact_path) if artifact_path else "",
        "stdout": out[-2000:],
        "stderr": err[-2000:],
        "readiness_checked": bool(not args.skip_readiness),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if code == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
