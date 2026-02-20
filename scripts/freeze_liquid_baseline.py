#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(raw: bytes) -> str:
    h = hashlib.sha256()
    h.update(raw)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_git(root: Path, args: List[str]) -> str:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(root), stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _git_snapshot(root: Path) -> Dict[str, object]:
    head = _run_git(root, ["rev-parse", "HEAD"])
    short_head = _run_git(root, ["rev-parse", "--short", "HEAD"])
    branch = _run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    status_raw = _run_git(root, ["status", "--porcelain"])
    dirty_lines = [ln for ln in status_raw.splitlines() if ln.strip()]
    return {
        "head": head,
        "head_short": short_head,
        "branch": branch,
        "dirty": bool(dirty_lines),
        "dirty_files_count": len(dirty_lines),
    }


def _to_repo_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path)


def _collect_model_artifacts(model_dir: Path, repo_root: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    missing_references: List[str] = []
    for p in sorted(model_dir.glob("*.json")):
        raw = p.read_bytes()
        sha = _sha256_file(p)
        rel = _to_repo_path(p, repo_root)
        item: Dict[str, object] = {
            "path": rel,
            "file_name": p.name,
            "size_bytes": int(p.stat().st_size),
            "sha256": sha,
        }
        try:
            payload = json.loads(raw.decode("utf-8"))
            item["model_name"] = str(payload.get("model_name") or "")
            item["model_version"] = str(payload.get("model_version") or "")
            item["type"] = str(payload.get("type") or "")
            item["feature_version"] = str(payload.get("feature_version") or "")
            item["data_version"] = str(payload.get("data_version") or "")
            item["feature_dim"] = int(payload.get("feature_dim") or 0) if payload.get("feature_dim") is not None else None
            components = payload.get("components")
            refs: List[str] = []
            if isinstance(components, dict):
                for _, v in components.items():
                    if isinstance(v, list):
                        refs.extend([str(x) for x in v if str(x).strip()])
            if refs:
                item["component_refs"] = refs
                for ref in refs:
                    target = model_dir / ref
                    if not target.exists():
                        missing_references.append(ref)
        except Exception:
            item["parse_error"] = "invalid_json"
        rows.append(item)
    return rows, sorted(set(missing_references))


def _effective_env(key: str, default: str) -> Dict[str, str]:
    val = os.getenv(key)
    return {
        "default": str(default),
        "effective": str(val if val is not None else default),
        "source": "env" if val is not None else "default",
    }


def _trainer_config_snapshot() -> Dict[str, Dict[str, str]]:
    return {
        "TRAIN_SEED": _effective_env("TRAIN_SEED", "42"),
        "LIQUID_EPOCHS": _effective_env("LIQUID_EPOCHS", "14"),
        "LIQUID_BATCH_SIZE": _effective_env("LIQUID_BATCH_SIZE", "128"),
        "TRAIN_GRAD_ACC_STEPS": _effective_env("TRAIN_GRAD_ACC_STEPS", "4"),
        "LIQUID_VAL_RATIO": _effective_env("LIQUID_VAL_RATIO", "0.2"),
        "LIQUID_EARLY_STOP_PATIENCE": _effective_env("LIQUID_EARLY_STOP_PATIENCE", "3"),
        "TRAIN_GRAD_CLIP": _effective_env("TRAIN_GRAD_CLIP", "1.0"),
        "LIQUID_WF_TRAIN_WINDOW": _effective_env("LIQUID_WF_TRAIN_WINDOW", "512"),
        "LIQUID_WF_TEST_WINDOW": _effective_env("LIQUID_WF_TEST_WINDOW", "96"),
        "LIQUID_WF_PURGE_WINDOW": _effective_env("LIQUID_WF_PURGE_WINDOW", "12"),
        "LIQUID_WF_STEP_WINDOW": _effective_env("LIQUID_WF_STEP_WINDOW", "96"),
        "LIQUID_PURGED_KFOLD_SPLITS": _effective_env("LIQUID_PURGED_KFOLD_SPLITS", "5"),
        "LIQUID_PURGED_KFOLD_PURGE": _effective_env("LIQUID_PURGED_KFOLD_PURGE", "12"),
        "LIQUID_MIN_TRAIN_POINTS": _effective_env("LIQUID_MIN_TRAIN_POINTS", "64"),
        "LIQUID_MIN_TEST_POINTS": _effective_env("LIQUID_MIN_TEST_POINTS", "16"),
        "VALIDATION_WF_TRAIN_WINDOW": _effective_env("VALIDATION_WF_TRAIN_WINDOW", "512"),
        "VALIDATION_WF_TEST_WINDOW": _effective_env("VALIDATION_WF_TEST_WINDOW", "96"),
        "VALIDATION_WF_PURGE_WINDOW": _effective_env("VALIDATION_WF_PURGE_WINDOW", "12"),
        "VALIDATION_WF_STEP_WINDOW": _effective_env("VALIDATION_WF_STEP_WINDOW", "96"),
        "VALIDATION_WF_MIN_FOLDS": _effective_env("VALIDATION_WF_MIN_FOLDS", "3"),
        "VALIDATION_PURGED_KFOLD_SPLITS": _effective_env("VALIDATION_PURGED_KFOLD_SPLITS", "5"),
        "VALIDATION_PURGED_KFOLD_PURGE": _effective_env("VALIDATION_PURGED_KFOLD_PURGE", "12"),
        "VALIDATION_MIN_TRAIN_POINTS": _effective_env("VALIDATION_MIN_TRAIN_POINTS", "64"),
        "VALIDATION_MIN_TEST_POINTS": _effective_env("VALIDATION_MIN_TEST_POINTS", "16"),
        "SOCIAL_POST_LATENT_DIM": _effective_env("SOCIAL_POST_LATENT_DIM", "32"),
        "SOCIAL_COMMENT_LATENT_DIM": _effective_env("SOCIAL_COMMENT_LATENT_DIM", "32"),
        "SOCIAL_AGG_BLEND_ALPHA": _effective_env("SOCIAL_AGG_BLEND_ALPHA", "0.35"),
        "LIQUID_TEXT_DROPOUT_PROB": _effective_env("LIQUID_TEXT_DROPOUT_PROB", "0.1"),
        "MULTIMODAL_TEXT_DROPOUT_PROB": _effective_env("MULTIMODAL_TEXT_DROPOUT_PROB", "0.1"),
        "FEATURE_VERSION": _effective_env("FEATURE_VERSION", "feature-store-v2.1"),
        "DATA_VERSION": _effective_env("DATA_VERSION", "v1"),
        "LIQUID_SYMBOLS": _effective_env("LIQUID_SYMBOLS", "BTC,ETH,SOL"),
        "LIQUID_PRIMARY_TIMEFRAME": _effective_env("LIQUID_PRIMARY_TIMEFRAME", "5m"),
        "TRAIN_DQ_GATE_MODE": _effective_env("TRAIN_DQ_GATE_MODE", "soft"),
        "LIQUID_DQ_GATE_MODE": _effective_env("LIQUID_DQ_GATE_MODE", "soft"),
    }


def _multimodal_baseline_snapshot() -> Dict[str, Dict[str, str]]:
    return {
        "MULTIMODAL_L2": _effective_env("MULTIMODAL_L2", "0.05"),
        "MULTIMODAL_TARGET_HORIZON_STEPS": _effective_env("MULTIMODAL_TARGET_HORIZON_STEPS", "1"),
        "MULTIMODAL_FUSION_MODE": _effective_env("MULTIMODAL_FUSION_MODE", "single_ridge"),
        "FEATURE_PAYLOAD_SCHEMA_VERSION": _effective_env("FEATURE_PAYLOAD_SCHEMA_VERSION", "main"),
    }


def _feature_contract_snapshot(repo_root: Path) -> Dict[str, object]:
    import sys

    infer_dir = repo_root / "inference"
    if str(infer_dir) not in sys.path:
        sys.path.append(str(infer_dir))
    try:
        from liquid_feature_contract import (  # type: ignore
            BASE_V2_FEATURE_KEYS,
            LIQUID_FEATURE_KEYS,
            LIQUID_FEATURE_SCHEMA_VERSION,
            LIQUID_FULL_FEATURE_KEYS,
            LIQUID_LATENT_FEATURE_KEYS,
            LIQUID_MANUAL_FEATURE_KEYS,
            ONLINE_LIQUID_FEATURE_KEYS,
        )

        return {
            "feature_payload_schema_version": str(LIQUID_FEATURE_SCHEMA_VERSION),
            "online_dim": int(len(ONLINE_LIQUID_FEATURE_KEYS)),
            "base_v2_dim": int(len(BASE_V2_FEATURE_KEYS)),
            "manual_dim": int(len(LIQUID_MANUAL_FEATURE_KEYS)),
            "latent_dim": int(len(LIQUID_LATENT_FEATURE_KEYS)),
            "full_dim": int(len(LIQUID_FULL_FEATURE_KEYS)),
            "effective_training_dim": int(len(LIQUID_FEATURE_KEYS)),
            "source": "module_import",
        }
    except Exception:
        src_path = infer_dir / "liquid_feature_contract.py"
        src = src_path.read_text(encoding="utf-8")
        base_block_match = re.search(r"BASE_V2_FEATURE_KEYS\s*:\s*List\[str\]\s*=\s*\[(.*?)\]\n", src, flags=re.S)
        base_dim = 0
        if base_block_match:
            base_dim = len(re.findall(r"\"[^\"]+\"", base_block_match.group(1)))
        manual_extra = 351
        latent_dim = 128
        m1 = re.search(r"manual_stat_\{i:03d\}\"\s+for\s+i\s+in\s+range\((\d+)\)", src)
        if m1:
            manual_extra = int(m1.group(1))
        m2 = re.search(r"latent_\{i:03d\}\"\s+for\s+i\s+in\s+range\((\d+)\)", src)
        if m2:
            latent_dim = int(m2.group(1))
        manual_dim = int(base_dim + manual_extra)
        full_dim = int(manual_dim + latent_dim)
        schema = str(os.getenv("FEATURE_PAYLOAD_SCHEMA_VERSION", "main"))
        schema_norm = schema.strip().lower()
        effective_dim = int(base_dim if schema_norm in {"v2", "v2.3", "legacy"} else full_dim)
        return {
            "feature_payload_schema_version": schema,
            "online_dim": int(base_dim),
            "base_v2_dim": int(base_dim),
            "manual_dim": manual_dim,
            "latent_dim": int(latent_dim),
            "full_dim": full_dim,
            "effective_training_dim": effective_dim,
            "source": "static_parse_fallback",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze current liquid baseline config and model manifest")
    parser.add_argument("--label", default="phase_a_baseline_freeze")
    parser.add_argument("--model-dir", default="backend/models")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = (repo_root / str(args.model_dir)).resolve()
    if not model_dir.exists():
        raise SystemExit(f"model_dir_not_found:{model_dir}")

    out_path = str(args.out).strip()
    if not out_path:
        out_path = f"baseline_snapshots/liquid_baseline_freeze_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_file = (repo_root / out_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    model_artifacts, missing_refs = _collect_model_artifacts(model_dir, repo_root)
    payload: Dict[str, object] = {
        "label": str(args.label),
        "generated_at": _utcnow_iso(),
        "scope": {"track": "liquid", "purpose": "baseline_freeze"},
        "git": _git_snapshot(repo_root),
        "feature_contract": _feature_contract_snapshot(repo_root),
        "trainer_defaults": _trainer_config_snapshot(),
        "multimodal_defaults": _multimodal_baseline_snapshot(),
        "model_dir": _to_repo_path(model_dir, repo_root),
        "model_artifacts": model_artifacts,
        "integrity_checks": {
            "artifact_count": len(model_artifacts),
            "missing_component_refs": missing_refs,
            "all_component_refs_present": len(missing_refs) == 0,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    out_obj = {
        "snapshot": payload,
        "snapshot_sha256": _sha256_bytes(canonical),
    }
    output_file.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_file = output_file.parent / "liquid_baseline_latest.json"
    latest_file.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "out": str(output_file.relative_to(repo_root)),
                "latest": str(latest_file.relative_to(repo_root)),
                "artifacts": len(model_artifacts),
                "missing_component_refs": len(missing_refs),
                "snapshot_sha256": out_obj["snapshot_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
