#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _latest_json(path: Path, prefix: str) -> Path | None:
    files = sorted(path.glob(f"{prefix}_*.json"))
    return files[-1] if files else None


def _read_json(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_snapshot(root: Path) -> str:
    phase_dir = root / "artifacts" / "phase63"
    hm = _read_json(_latest_json(phase_dir, "hard_metrics"))
    parity = _read_json(_latest_json(phase_dir, "parity"))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    track = hm.get("track", "liquid")
    score_source = hm.get("score_source", "model")
    sharpe = hm.get("sharpe", "n/a")
    maxdd = hm.get("max_drawdown", "n/a")
    reject = hm.get("execution_reject_rate", "n/a")
    hard = hm.get("hard_passed", hm.get("passed", False))
    parity_status = parity.get("status", "unknown")
    matched = parity.get("matched_targets_count", 0)
    orders = parity.get("paper_filled_orders_count", 0)
    return (
        f"### Auto Snapshot ({ts})\n"
        f"- track: `{track}`\n"
        f"- score_source: `{score_source}`\n"
        f"- sharpe: `{sharpe}`\n"
        f"- max_drawdown: `{maxdd}`\n"
        f"- execution_reject_rate: `{reject}`\n"
        f"- hard_passed: `{str(bool(hard)).lower()}`\n"
        f"- parity_status: `{parity_status}`\n"
        f"- parity_matched_targets: `{matched}`\n"
        f"- parity_paper_filled_orders: `{orders}`\n"
    )


def _inject_between_markers(content: str, section: str, begin: str, end: str) -> str:
    if begin not in content or end not in content:
        return content.rstrip() + "\n\n" + begin + "\n" + section.rstrip() + "\n" + end + "\n"
    left, rest = content.split(begin, 1)
    _, right = rest.split(end, 1)
    return left + begin + "\n" + section.rstrip() + "\n" + end + right


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate and inject status snapshot into README/TRACKING")
    ap.add_argument("--root", default=".")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    snapshot = _build_snapshot(root)
    print(snapshot)
    if not args.write:
        return 0

    begin = "<!-- AUTO_STATUS_SNAPSHOT:BEGIN -->"
    end = "<!-- AUTO_STATUS_SNAPSHOT:END -->"
    for rel in ("README.md", "TRACKING.md"):
        p = root / rel
        if not p.exists():
            continue
        content = p.read_text(encoding="utf-8")
        updated = _inject_between_markers(content, snapshot, begin, end)
        p.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
