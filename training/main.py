"""Strict training entrypoint: VC + Liquid sequence models only."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


def _env_flag(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"training_subprocess_failed:{cmd}:{proc.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Training Service V2 strict entrypoint")
    ap.add_argument("--train-vc", action="store_true", default=_env_flag("TRAIN_ENABLE_VC", "1"))
    ap.add_argument("--train-liquid", action="store_true", default=_env_flag("TRAIN_ENABLE_LIQUID", "1"))
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    py = str(args.python)
    if bool(args.train_vc):
        _run([py, "training/train_vc.py"])
    if bool(args.train_liquid):
        _run([py, "training/train_liquid.py"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
