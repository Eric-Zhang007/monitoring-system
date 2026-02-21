#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.append(str(ROOT / "backend"))

from liquid_model_registry import rollback_active  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Rollback liquid active model by symbol+horizon")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--horizon", default="1h", choices=["1h", "4h", "1d", "7d"])
    ap.add_argument("--operator", default="manual")
    args = ap.parse_args()

    out = rollback_active(
        symbol=str(args.symbol).strip().upper(),
        horizon=str(args.horizon).strip().lower(),
        actor=str(args.operator),
    )
    print(json.dumps({"status": "ok", **out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
