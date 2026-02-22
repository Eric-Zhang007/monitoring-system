from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "backend", ROOT / "inference", ROOT / "training"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
