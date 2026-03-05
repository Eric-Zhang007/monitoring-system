#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Set

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from features.feature_contract import SCHEMA_HASH
from training.cache.panel_cache import CacheBuildConfig, build_training_cache_from_db


def _parse_ts(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        return datetime.now(timezone.utc)
    text = text.replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_symbols(items: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in list(items or []):
        sym = str(raw or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _blocked_symbols_from_readiness(payload: dict) -> Set[str]:
    blocked = _normalize_symbols(payload.get("blocked_symbols") or [])
    if blocked:
        return set(blocked)
    rows = payload.get("symbol_readiness")
    if not isinstance(rows, dict):
        return set()
    out: List[str] = []
    for sym, row in rows.items():
        status = str((row or {}).get("status") or "").strip().upper()
        if status == "BLOCKED":
            out.append(str(sym))
    return set(_normalize_symbols(out))


def main() -> int:
    ap = argparse.ArgumentParser(description="Build strict top50 panel training cache (no per-sample DB reads)")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--universe-snapshot", default=os.getenv("LIQUID_UNIVERSE_SNAPSHOT_FILE", "artifacts/universe/liquid_top50_snapshot.json"))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--bar-size", default=os.getenv("LIQUID_CACHE_BAR_SIZE", "5m"))
    # Keep cache build default aligned with training default.
    ap.add_argument("--lookback-len", type=int, default=int(os.getenv("LIQUID_LOOKBACK", "2016")))
    ap.add_argument("--horizons", default=os.getenv("LIQUID_HORIZONS", "1h,4h,1d,7d"))
    ap.add_argument("--feature-contract-hash", default=os.getenv("LIQUID_FEATURE_CONTRACT_HASH", SCHEMA_HASH))
    ap.add_argument("--output-dir", default=os.getenv("LIQUID_TRAIN_CACHE_DIR", "artifacts/cache/liquid_top50"))
    ap.add_argument("--incremental", type=int, default=int(os.getenv("LIQUID_CACHE_INCREMENTAL", "1")))
    ap.add_argument(
        "--incremental-warmup-steps",
        type=int,
        default=int(os.getenv("LIQUID_CACHE_INCREMENTAL_WARMUP_STEPS", "288")),
    )
    ap.add_argument(
        "--readiness-file",
        default=os.getenv("LIQUID_DATA_READINESS_FILE", "artifacts/audit/top50_data_readiness_latest.json"),
    )
    ap.add_argument(
        "--exclude-blocked",
        type=int,
        default=int(os.getenv("LIQUID_EXCLUDE_BLOCKED_SYMBOLS", "1")),
    )
    ap.add_argument(
        "--context-timeframes",
        default=os.getenv("ANALYST_CONTEXT_TIMEFRAMES", "5m,15m,1h,4h,1d"),
    )
    ap.add_argument(
        "--require-multi-tf-context",
        type=int,
        default=int(os.getenv("LIQUID_REQUIRE_MULTI_TF_CONTEXT", "1")),
    )
    args = ap.parse_args()

    universe_snapshot_path = Path(str(args.universe_snapshot))
    if not universe_snapshot_path.exists():
        raise RuntimeError(f"universe_snapshot_missing:{universe_snapshot_path}")
    universe_payload = json.loads(universe_snapshot_path.read_text(encoding="utf-8"))
    symbols = _normalize_symbols(universe_payload.get("symbols") or [])
    if not symbols:
        raise RuntimeError("universe_snapshot_symbols_empty")

    excluded_symbols: List[str] = []
    if bool(int(args.exclude_blocked)):
        readiness_path = Path(str(args.readiness_file))
        if not readiness_path.exists():
            raise RuntimeError(f"readiness_file_missing:{readiness_path}")
        readiness_payload = json.loads(readiness_path.read_text(encoding="utf-8"))
        blocked = _blocked_symbols_from_readiness(readiness_payload)
        if blocked:
            symbols = [s for s in symbols if s not in blocked]
            excluded_symbols = sorted(list(blocked))
        if not symbols:
            raise RuntimeError("all_universe_symbols_blocked_by_readiness")

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_universe_path = output_dir / "universe_snapshot.filtered.json"
    filtered_payload = dict(universe_payload)
    filtered_payload["symbols"] = symbols
    filtered_payload["filtered_by_readiness"] = {
        "enabled": bool(int(args.exclude_blocked)),
        "readiness_file": str(args.readiness_file),
        "excluded_symbols": excluded_symbols,
    }
    filtered_universe_path.write_text(json.dumps(filtered_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cfg = CacheBuildConfig(
        universe_snapshot_file=filtered_universe_path,
        start_ts=_parse_ts(str(args.start)),
        end_ts=_parse_ts(str(args.end)),
        bar_size=str(args.bar_size).strip().lower() or "5m",
        lookback_len=max(8, int(args.lookback_len)),
        horizons=[s.strip().lower() for s in str(args.horizons).split(",") if s.strip()],
        feature_contract_hash=str(args.feature_contract_hash),
        output_dir=output_dir,
        database_url=str(args.database_url),
        incremental=bool(int(args.incremental)),
        incremental_warmup_steps=max(8, int(args.incremental_warmup_steps)),
        context_timeframes=[s.strip().lower() for s in str(args.context_timeframes).split(",") if s.strip()],
        require_multi_tf_context=bool(int(args.require_multi_tf_context)),
    )
    if not cfg.horizons:
        raise RuntimeError("cache_horizons_empty")

    manifest = build_training_cache_from_db(cfg)
    idx_npz = np.load(cfg.output_dir / "index.npz")
    print(
        json.dumps(
            {
                "status": "ok",
                "cache_dir": str(cfg.output_dir),
                "manifest_file": str(cfg.output_dir / "cache_manifest.json"),
                "audit_file": str(cfg.output_dir / "data_audit.json"),
                "cache_hash": manifest.get("cache_hash"),
                "sample_count": int(len(idx_npz["t_idx"])),
                "excluded_symbols": excluded_symbols,
                "incremental": bool(int(args.incremental)),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
