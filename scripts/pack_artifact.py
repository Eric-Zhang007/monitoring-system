#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cost.cost_profile import cost_profile_snapshot
from mlops_artifacts.validate import validate_manifest_dir
from schema.schema_hash import compute_schema_hash


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack strict liquid artifact with full manifest and hashes")
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--universe-snapshot", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--cost-profile", default="standard")
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    src = Path(str(args.artifact_dir))
    validate_manifest_dir(src)
    out = Path(str(args.output_dir)).resolve() if str(args.output_dir).strip() else src.resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest_src = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    files = manifest_src.get("files") or {}
    for key in ("weights", "schema_snapshot", "training_report"):
        rel = str(files.get(key) or "")
        if not rel:
            raise RuntimeError(f"manifest_files_entry_missing:{key}")
        src_file = src / rel
        if not src_file.exists():
            raise RuntimeError(f"artifact_file_missing:{key}:{src_file}")
        dst_file = out / src_file.name
        if src_file.resolve() != dst_file.resolve():
            shutil.copyfile(src_file, dst_file)

    weights_path = out / str(files["weights"])
    ckpt = torch.load(weights_path, map_location="cpu")
    universe_path = Path(str(args.universe_snapshot))
    if not universe_path.exists():
        raise RuntimeError(f"universe_snapshot_missing:{universe_path}")
    universe_payload = json.loads(universe_path.read_text(encoding="utf-8"))

    schema_ext = {
        "feature_contract_hash": str(ckpt.get("schema_hash") or ""),
        "universe_snapshot_hash": str(universe_payload.get("snapshot_hash") or ""),
        "symbol_to_id": dict(ckpt.get("symbol_to_id") or {}),
        "cache_hash": str(ckpt.get("cache_hash") or ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    schema_ext_path = out / "schema_snapshot_ext.json"
    schema_ext_path.write_text(json.dumps(schema_ext, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    eval_dir = Path(str(args.eval_dir))
    if not eval_dir.exists():
        raise RuntimeError(f"eval_dir_missing:{eval_dir}")
    eval_report = out / "evaluation_report.md"
    if (eval_dir / "report.md").exists():
        shutil.copyfile(eval_dir / "report.md", eval_report)
    else:
        eval_report.write_text("# Evaluation Report\n\nmissing source report.md\n", encoding="utf-8")

    router_report = out / "router_report.json"
    if (eval_dir / "router_report.json").exists():
        shutil.copyfile(eval_dir / "router_report.json", router_report)
    else:
        router_report.write_text(json.dumps({"status": "missing"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cost_path = out / "cost_profile_snapshot.json"
    cost_path.write_text(json.dumps(cost_profile_snapshot(str(args.cost_profile)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    schema_snapshot = out / str(files["schema_snapshot"])
    schema_hash = compute_schema_hash(schema_snapshot)
    final_files = {
        "weights": str(weights_path.name),
        "schema_snapshot": str(schema_snapshot.name),
        "training_report": str(files["training_report"]),
        "schema_snapshot_ext": str(schema_ext_path.name),
        "cost_profile_snapshot": str(cost_path.name),
        "evaluation_report": str(eval_report.name),
        "router_report": str(router_report.name),
    }
    hash_map: Dict[str, str] = {}
    for rel in final_files.values():
        hash_map[rel] = _hash_file(out / rel)
    manifest = {
        "model_id": str(manifest_src.get("model_id") or "liquid_main"),
        "schema_hash": str(schema_hash),
        "lookback": int(manifest_src.get("lookback") or 0),
        "feature_dim": int(manifest_src.get("feature_dim") or 0),
        "data_version": str(manifest_src.get("data_version") or "main"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics_summary": dict(manifest_src.get("metrics_summary") or {}),
        "required_files": ["schema_snapshot_ext", "cost_profile_snapshot", "router_report"],
        "files": final_files,
        "file_hashes": hash_map,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"status": "ok", "manifest": str(out / "manifest.json"), "output_dir": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
