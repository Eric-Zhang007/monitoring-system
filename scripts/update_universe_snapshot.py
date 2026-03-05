#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).resolve().parent / "update_universe_top50.py"
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(target),
            *sys.argv[1:],
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

