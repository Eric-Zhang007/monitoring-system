#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate training cache data_audit gate")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--max-missing-ratio", type=float, default=0.25)
    args = ap.parse_args()

    audit_path = Path(str(args.cache_dir)) / "data_audit.json"
    if not audit_path.exists():
        raise RuntimeError(f"cache_audit_missing:{audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    leak_ok = bool((audit.get("asof_leakage_check") or {}).get("passed", False))
    if not leak_ok:
        raise RuntimeError("cache_leakage_check_failed")
    miss = float((audit.get("coverage") or {}).get("avg_missing_ratio", audit.get("missing_ratio", 1.0)) or 1.0)
    if miss > float(args.max_missing_ratio):
        raise RuntimeError(f"cache_missing_ratio_too_high:{miss:.6f}:{float(args.max_missing_ratio):.6f}")
    print(json.dumps({"status": "ok", "missing_ratio": miss, "leakage_passed": leak_ok}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
