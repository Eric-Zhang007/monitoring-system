from __future__ import annotations

import sys
from pathlib import Path


def activate() -> None:
    root = Path(__file__).resolve().parents[1]
    for p in (root, root / "backend", root / "inference", root / "training"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
