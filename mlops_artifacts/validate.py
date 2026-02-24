from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from schema.schema_hash import compute_schema_hash


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"manifest_parse_failed:{path}:{exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"manifest_invalid_type:{path}")
    return payload


def validate_manifest_dir(model_dir: str | Path, *, expected_schema_hash: Optional[str] = None) -> Dict[str, Any]:
    root = Path(model_dir)
    if not root.exists():
        raise RuntimeError(f"artifact_dir_missing:{root}")
    if not root.is_dir():
        raise RuntimeError(f"artifact_dir_not_directory:{root}")

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"manifest_missing:{manifest_path}")
    manifest = _load_json(manifest_path)

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("manifest_files_missing")
    required_files = ("weights", "schema_snapshot", "training_report")
    for key in required_files:
        rel = files.get(key)
        if not isinstance(rel, str) or not rel.strip():
            raise RuntimeError(f"manifest_files_entry_missing:{key}")
        fpath = root / rel
        if not fpath.exists():
            raise RuntimeError(f"artifact_file_missing:{key}:{fpath}")

    schema_snapshot = root / str(files["schema_snapshot"])
    actual_hash = compute_schema_hash(schema_snapshot)
    manifest_hash = str(manifest.get("schema_hash") or "")
    if not manifest_hash:
        raise RuntimeError("manifest_schema_hash_missing")
    if actual_hash != manifest_hash:
        raise RuntimeError(f"manifest_schema_hash_mismatch:{actual_hash}:{manifest_hash}")
    if expected_schema_hash is not None and str(expected_schema_hash) != manifest_hash:
        raise RuntimeError(f"schema_hash_mismatch_expected:{expected_schema_hash}:{manifest_hash}")

    return manifest

