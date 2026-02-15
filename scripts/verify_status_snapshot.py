#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import generate_status_snapshot as g

BEGIN = "<!-- AUTO_STATUS_SNAPSHOT:BEGIN -->"
END = "<!-- AUTO_STATUS_SNAPSHOT:END -->"


def _extract(content: str) -> str | None:
    if BEGIN not in content or END not in content:
        return None
    left = content.split(BEGIN, 1)[1]
    body = left.split(END, 1)[0]
    return body.strip()


def _normalize(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].startswith("### Auto Snapshot"):
        lines = lines[1:]
    return "\n".join(lines)


def main() -> int:
    root = Path('.').resolve()
    expected = _normalize(g._build_snapshot(root).strip())
    ok = True
    for rel in ("README.md", "TRACKING.md"):
        p = root / rel
        if not p.exists():
            continue
        content = p.read_text(encoding='utf-8')
        got = _extract(content)
        if got is None:
            print(f"missing markers in {rel}")
            ok = False
            continue
        if _normalize(got) != expected:
            print(f"snapshot mismatch in {rel}")
            ok = False
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
