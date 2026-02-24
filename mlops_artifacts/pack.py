from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from schema.schema_hash import compute_schema_hash


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def pack_model_artifact(
    *,
    model_dir: str | Path,
    model_id: str,
    state_dict: Dict[str, Any],
    schema_path: str | Path,
    lookback: int,
    feature_dim: int,
    data_version: str,
    metrics_summary: Dict[str, Any],
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)

    schema_src = Path(schema_path)
    if not schema_src.exists():
        raise RuntimeError(f"schema_missing:{schema_src}")

    schema_snapshot = out / "schema_snapshot.yaml"
    shutil.copyfile(schema_src, schema_snapshot)
    schema_hash = compute_schema_hash(schema_snapshot)

    weights_path = out / "weights.pt"
    torch.save(state_dict, weights_path)

    training_report_path = out / "training_report.json"
    report_payload = dict(extra_payload or {})
    training_report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest: Dict[str, Any] = {
        "model_id": str(model_id),
        "schema_hash": str(schema_hash),
        "lookback": int(lookback),
        "feature_dim": int(feature_dim),
        "data_version": str(data_version),
        "created_at": _utcnow_iso(),
        "metrics_summary": dict(metrics_summary or {}),
        "files": {
            "weights": weights_path.name,
            "schema_snapshot": schema_snapshot.name,
            "training_report": training_report_path.name,
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest

